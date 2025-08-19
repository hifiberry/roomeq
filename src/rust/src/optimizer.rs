use crate::filters::*;
use serde::{Deserialize, Serialize};

/// Complete optimization job input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationJob {
    pub measured_curve: FrequencyResponse,
    pub target_curve: TargetCurveData,
    pub optimizer_params: OptimizerPreset,
    pub sample_rate: f64,
    pub filter_count: usize,
}

/// Target curve data with points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCurveData {
    pub name: String,
    pub description: String,
    pub expert: bool,
    pub curve: Vec<CurvePoint>,
}

/// Step output during optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    pub step: usize,
    pub filters: Vec<BiquadFilter>,
    pub corrected_response: FrequencyResponse,
    pub residual_error: f64,
    pub message: String,
    pub progress_percent: f64,
}

/// Complete optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub success: bool,
    pub filters: Vec<BiquadFilter>,
    pub final_error: f64,
    pub original_error: f64,
    pub improvement_db: f64,
    pub steps: Vec<OptimizationStep>,
    pub processing_time_ms: u64,
    pub error_message: Option<String>,
}

pub struct RoomEQOptimizer {
    sample_rate: f64,
    min_frequency: f64,
    max_frequency: f64,
}

impl RoomEQOptimizer {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            min_frequency: 20.0,
            max_frequency: 20000.0,
        }
    }

    /// Generate target response from curve definition
    pub fn generate_target_response(&self, target_curve: &TargetCurveData, frequencies: &[f64]) -> Vec<f64> {
        let mut target_response = Vec::with_capacity(frequencies.len());

        for &freq in frequencies {
            let target_db = self.interpolate_target_curve(target_curve, freq);
            target_response.push(target_db);
        }

        target_response
    }

    fn interpolate_target_curve(&self, target_curve: &TargetCurveData, frequency: f64) -> f64 {
        let curve_points = &target_curve.curve;
        
        if curve_points.is_empty() {
            return 0.0;
        }

        if frequency <= curve_points[0].frequency {
            return curve_points[0].target_db;
        }

        if frequency >= curve_points[curve_points.len() - 1].frequency {
            return curve_points[curve_points.len() - 1].target_db;
        }

        // Find the two points to interpolate between
        let mut i = 0;
        while i < curve_points.len() - 1 && curve_points[i + 1].frequency < frequency {
            i += 1;
        }

        let p1 = &curve_points[i];
        let p2 = &curve_points[i + 1];

        // Linear interpolation in log-frequency space
        let log_f = frequency.ln();
        let log_f1 = p1.frequency.ln();
        let log_f2 = p2.frequency.ln();
        
        let t = (log_f - log_f1) / (log_f2 - log_f1);
        p1.target_db + t * (p2.target_db - p1.target_db)
    }

    /// Calculate RMS error between measured and target responses
    pub fn calculate_error(&self, measured: &[f64], target: &[f64]) -> f64 {
        assert_eq!(measured.len(), target.len());
        
        let mut sum_sq = 0.0;
        let mut count = 0;

        for i in 0..measured.len() {
            let error = measured[i] - target[i];
            sum_sq += error * error;
            count += 1;
        }

        if count > 0 {
            (sum_sq / count as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Generate optimization frequencies (log-spaced)
    pub fn generate_frequencies(&self, num_points: usize) -> Vec<f64> {
        let log_min = self.min_frequency.ln();
        let log_max = self.max_frequency.ln();
        let step = (log_max - log_min) / (num_points - 1) as f64;

        (0..num_points)
            .map(|i| (log_min + i as f64 * step).exp())
            .collect()
    }

    /// Optimize EQ filters using a simplified approach
    pub fn optimize(&self, job: OptimizationJob) -> OptimizationResult {
        let start_time = std::time::Instant::now();
        let mut steps = Vec::new();
        
        // Generate optimization frequencies
        let frequencies = self.generate_frequencies(200);
        
        // Generate target response
        let target_response = self.generate_target_response(&job.target_curve, &frequencies);
        
        // Interpolate measured response to optimization frequencies
        let measured_response: Vec<f64> = frequencies.iter()
            .map(|&freq| job.measured_curve.interpolate_at(freq))
            .collect();

        // Calculate original error
        let original_error = self.calculate_error(&measured_response, &target_response);
        
        let mut filters = Vec::new();
        let mut current_response = measured_response.clone();

        // Add high-pass filter if requested
        if job.optimizer_params.add_highpass {
            let hp_filter = BiquadFilter::high_pass(80.0, 0.707, self.sample_rate);
            let hp_response = hp_filter.frequency_response(&frequencies, self.sample_rate);
            
            // Apply high-pass response
            for i in 0..current_response.len() {
                current_response[i] += hp_response[i];
            }
            
            filters.push(hp_filter);
            
            let error = self.calculate_error(&current_response, &target_response);
            let step = OptimizationStep {
                step: 0,
                filters: filters.clone(),
                corrected_response: FrequencyResponse::new(frequencies.clone(), current_response.clone()),
                residual_error: error,
                message: "Added high-pass filter".to_string(),
                progress_percent: 0.0,
            };
            steps.push(step.clone());
            println!("{}", serde_json::to_string(&step).unwrap());
        }

        // Simple optimization: add filters for largest errors
        for filter_idx in 0..job.filter_count {
            let progress = (filter_idx as f64 / job.filter_count as f64) * 100.0;
            
            // Find frequency with largest error
            let mut max_error = 0.0;
            let mut max_error_freq = 1000.0;
            let mut max_error_idx = 0;

            for i in 0..current_response.len() {
                let error = (current_response[i] - target_response[i]).abs();
                if error > max_error && frequencies[i] >= 20.0 && frequencies[i] <= 20000.0 {
                    max_error = error;
                    max_error_freq = frequencies[i];
                    max_error_idx = i;
                }
            }

            if max_error < 0.1 {
                break; // Error is small enough
            }

            // Determine gain needed
            let gain_needed = target_response[max_error_idx] - current_response[max_error_idx];
            let gain_db = gain_needed.max(job.optimizer_params.mindb).min(job.optimizer_params.maxdb);
            
            // Determine Q factor (limited by qmax)
            let q = (job.optimizer_params.qmax * 0.3).max(1.0).min(job.optimizer_params.qmax);

            // Create appropriate filter type based on frequency and gain
            let new_filter = if max_error_freq < 200.0 && gain_db > 0.0 {
                BiquadFilter::low_shelf(max_error_freq, q, gain_db, self.sample_rate)
            } else if max_error_freq > 10000.0 && gain_db < 0.0 {
                BiquadFilter::high_shelf(max_error_freq, q, gain_db, self.sample_rate)
            } else {
                BiquadFilter::peaking_eq(max_error_freq, q, gain_db, self.sample_rate)
            };

            // Apply filter response
            let filter_response = new_filter.frequency_response(&frequencies, self.sample_rate);
            for i in 0..current_response.len() {
                current_response[i] += filter_response[i];
            }

            filters.push(new_filter);

            let error = self.calculate_error(&current_response, &target_response);
            let step = OptimizationStep {
                step: filter_idx + 1,
                filters: filters.clone(),
                corrected_response: FrequencyResponse::new(frequencies.clone(), current_response.clone()),
                residual_error: error,
                message: format!("Added filter {} at {:.1}Hz ({:.1}dB)", filter_idx + 1, max_error_freq, gain_db),
                progress_percent: progress,
            };
            steps.push(step.clone());
            println!("{}", serde_json::to_string(&step).unwrap());
        }

        let final_error = self.calculate_error(&current_response, &target_response);
        let improvement_db = 20.0 * (original_error / final_error.max(1e-10)).log10();
        let processing_time = start_time.elapsed().as_millis() as u64;

        OptimizationResult {
            success: true,
            filters,
            final_error,
            original_error,
            improvement_db,
            steps,
            processing_time_ms: processing_time,
            error_message: None,
        }
    }
}
