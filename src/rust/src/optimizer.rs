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
    pub usable_freq_low: f64,
    pub usable_freq_high: f64,
}

pub struct RoomEQOptimizer {
    sample_rate: f64,
    min_frequency: f64,
    max_frequency: f64,
    output_progress: bool,
    human_readable: bool,
    output_frequency_response: bool,
}

impl RoomEQOptimizer {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            min_frequency: 20.0,
            max_frequency: 20000.0,
            output_progress: true,  // Default to current behavior
            human_readable: false,  // Default to JSON output
            output_frequency_response: false,  // Default to no frequency response
        }
    }

    /// Configure output mode for progress steps and format
    pub fn set_output_mode(&mut self, output_progress: bool, human_readable: bool, output_frequency_response: bool) {
        self.output_progress = output_progress;
        self.human_readable = human_readable;
        self.output_frequency_response = output_frequency_response;
    }

    /// Output an optimization step in the configured format
    /// Output optimization step with optional frequency response in the configured format
    fn output_step_with_frequency_response(&self, step: &OptimizationStep, frequencies: Option<&[f64]>, filter_magnitude: Option<&[f64]>, filter_phase: Option<&[f64]>, original_magnitude: Option<&[f64]>) {
        if self.human_readable {
            println!("Step {}: {} (Error: {:.2} dB, Progress: {:.1}%)", 
                     step.step, step.message, step.residual_error, step.progress_percent);
            
            if let Some(filter) = step.filters.last() {
                println!("  Added: {}", filter.as_text());
            }

            // Output frequency response if enabled and data provided
            if self.output_frequency_response && frequencies.is_some() && filter_magnitude.is_some() {
                let freq_data = frequencies.unwrap();
                let filter_mag_data = filter_magnitude.unwrap();
                println!("FREQUENCY_RESPONSE:step_{}:frequencies=[{:.1}-{:.1}Hz], filter_magnitudes=[{:.2}-{:.2}dB]", 
                         step.step, freq_data[0], freq_data[freq_data.len()-1],
                         filter_mag_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                         filter_mag_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            }
        } else {
            // JSON output - combine step data with frequency response if enabled
            let mut step_json = serde_json::to_value(step).unwrap();
            
            if self.output_frequency_response && frequencies.is_some() && filter_magnitude.is_some() {
                let frequencies = frequencies.unwrap();
                let filter_mag = filter_magnitude.unwrap();
                
                // Calculate resulting response (original + filter applied)
                let resulting_magnitude = if let Some(orig_mag) = original_magnitude {
                    orig_mag.iter().zip(filter_mag.iter())
                           .map(|(orig, filt)| orig + filt)
                           .collect::<Vec<f64>>()
                } else {
                    filter_mag.to_vec()
                };

                let frequency_response = serde_json::json!({
                    "frequencies": frequencies,
                    "filter_response": {
                        "magnitude_db": filter_mag,
                        "phase_degrees": filter_phase.unwrap_or(&[])
                    },
                    "resulting_response": {
                        "magnitude_db": resulting_magnitude
                    }
                });
                
                step_json["frequency_response"] = frequency_response;
            }
            
            if let Ok(json) = serde_json::to_string(&step_json) {
                println!("{}", json);
            }
        }
    }

    fn output_step(&self, step: &OptimizationStep) {
        self.output_step_with_frequency_response(step, None, None, None, None);
    }

    /// Calculate and output frequency response if enabled
    fn output_frequency_response(&self, filters: &[BiquadFilter], frequencies: &[f64]) {
        if !self.output_frequency_response {
            return;
        }

        let (magnitude_db, phase_degrees) = cascade_frequency_and_phase_response(filters, frequencies, self.sample_rate);
        
        if self.human_readable {
            println!("FREQUENCY_RESPONSE: final:frequencies=[{:.1}-{:.1}Hz], magnitudes=[{:.2}-{:.2}dB]", 
                     frequencies[0], frequencies[frequencies.len()-1],
                     magnitude_db.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                     magnitude_db.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        } else {
            let frequency_response = serde_json::json!({
                "event": "frequency_response_final",
                "frequencies": frequencies,
                "magnitude_db": magnitude_db,
                "phase_degrees": phase_degrees,
                "step": -1,
                "current_filter_set": filters
            });
            
            if let Ok(json) = serde_json::to_string(&frequency_response) {
                println!("{}", json);
            }
        }
    }

    /// Generate frequency candidates for optimization
    fn generate_frequency_candidates(&self, f_low: f64, f_high: f64, measured_frequencies: &[f64]) -> Vec<f64> {
        let mut frequencies = Vec::new();
        
        // First, add frequencies from the measured curve
        for &freq in measured_frequencies {
            if freq >= f_low && freq <= f_high {
                frequencies.push(freq);
            }
        }
        
        // Add interpolated frequencies between each pair of measured frequencies
        for i in 0..measured_frequencies.len() - 1 {
            let f1 = measured_frequencies[i];
            let f2 = measured_frequencies[i + 1];
            
            // Only add if both frequencies are in usable range
            if f1 >= f_low && f2 <= f_high && f1 < f2 {
                // Add geometric mean between f1 and f2
                let f_interp = (f1 * f2).sqrt();
                if f_interp > f_low && f_interp < f_high {
                    frequencies.push(f_interp);
                }
            }
        }
        
        // Add 1/3 octave frequencies in the usable range
        let mut freq = f_low;
        let max_freq = f_high;
        
        while freq <= max_freq {
            // Only add if not too close to existing frequencies
            let mut too_close = false;
            for &existing_freq in &frequencies {
                if (freq / existing_freq).abs().log2().abs() < 0.15 { // Less than ~1/6 octave apart
                    too_close = true;
                    break;
                }
            }
            
            if !too_close {
                frequencies.push(freq);
            }
            freq *= 2.0_f64.powf(1.0/3.0); // 1/3 octave steps
        }
        
        // Sort and remove frequencies too close to f_low or f_high boundaries for PEQ
        frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Remove frequencies at the exact boundaries (reserve for high-pass only)
        frequencies.retain(|&f| f > f_low * 1.05 && f < f_high * 0.95);
        
        // Ensure we have at least a few candidates
        if frequencies.is_empty() {
            let mid_freq = (f_low * f_high).sqrt();
            frequencies.push(mid_freq);
        }
        
        frequencies
    }

    /// Generate Q factor candidates
    fn generate_q_candidates(&self, params: &OptimizerPreset) -> Vec<f64> {
        let mut q_values = Vec::new();
        
        // Generate Q values from 0.5 to qmax in reasonable steps
        let q_min = 0.5;
        let q_max = params.qmax;
        let q_steps = 10; // Number of Q steps
        
        for i in 0..=q_steps {
            let t = i as f64 / q_steps as f64;
            // Use exponential distribution for Q values (more resolution at lower Q)
            let q = q_min * (q_max / q_min).powf(t);
            q_values.push(q);
        }
        
        q_values
    }

    /// Generate gain candidates in 0.5dB steps
    fn generate_gain_candidates(&self, params: &OptimizerPreset) -> Vec<f64> {
        let mut gains = Vec::new();
        
        let gain_min = params.mindb;
        let gain_max = params.maxdb;
        let step_size = 0.5; // 0.5dB steps as requested
        
        let mut gain = gain_min;
        while gain <= gain_max {
            gains.push(gain);
            gain += step_size;
        }
        
        gains
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

    /// Interpolate weight for a given frequency from target curve
    fn interpolate_weight(&self, target_curve: &TargetCurveData, frequency: f64, error_sign: f64) -> f64 {
        let curve_points = &target_curve.curve;
        
        if curve_points.is_empty() {
            return 1.0;
        }

        // Find the appropriate weight
        if frequency <= curve_points[0].frequency {
            return self.extract_weight_value(&curve_points[0].weight, error_sign);
        }

        if frequency >= curve_points[curve_points.len() - 1].frequency {
            return self.extract_weight_value(&curve_points[curve_points.len() - 1].weight, error_sign);
        }

        // Find the two points to interpolate between
        let mut i = 0;
        while i < curve_points.len() - 1 && curve_points[i + 1].frequency < frequency {
            i += 1;
        }

        let p1 = &curve_points[i];
        let p2 = &curve_points[i + 1];

        // Extract weights from both points
        let w1 = self.extract_weight_value(&p1.weight, error_sign);
        let w2 = self.extract_weight_value(&p2.weight, error_sign);

        // Linear interpolation in log-frequency space
        let log_f = frequency.ln();
        let log_f1 = p1.frequency.ln();
        let log_f2 = p2.frequency.ln();
        
        if (log_f2 - log_f1).abs() < 1e-10 {
            return w1;
        }

        let t = (log_f - log_f1) / (log_f2 - log_f1);
        w1 + t * (w2 - w1)
    }

    /// Extract weight value considering asymmetric weights (positive vs negative errors)
    fn extract_weight_value(&self, weight_opt: &Option<Weight>, error_sign: f64) -> f64 {
        match weight_opt {
            None => 1.0,
            Some(Weight::Single(w)) => *w,
            Some(Weight::Tuple(w_pos, w_neg)) => {
                if error_sign >= 0.0 { *w_pos } else { *w_neg }
            },
            Some(Weight::Triple(w_pos, w_neg, _)) => {
                if error_sign >= 0.0 { *w_pos } else { *w_neg }
            }
        }
    }

    /// Generate target weights for optimization frequencies
    pub fn generate_target_weights(&self, target_curve: &TargetCurveData, frequencies: &[f64], 
                                 measured: &[f64], target: &[f64]) -> Vec<f64> {
        let mut weights = Vec::with_capacity(frequencies.len());

        for i in 0..frequencies.len() {
            let freq = frequencies[i];
            let error = measured[i] - target[i];
            let weight = self.interpolate_weight(target_curve, freq, error);
            weights.push(weight);
        }

        weights
    }

    /// Calculate RMS error between measured and target responses
    #[allow(dead_code)]
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

    /// Calculate weighted RMS error between measured and target responses
    pub fn calculate_weighted_error(&self, measured: &[f64], target: &[f64], weights: &[f64]) -> f64 {
        assert_eq!(measured.len(), target.len());
        assert_eq!(measured.len(), weights.len());
        
        let mut weighted_sum_sq = 0.0;
        let mut total_weight = 0.0;

        for i in 0..measured.len() {
            let error = measured[i] - target[i];
            let weight = weights[i];
            weighted_sum_sq += (error * error) * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            (weighted_sum_sq / total_weight).sqrt()
        } else {
            0.0
        }
    }

    /// Find usable frequency range for optimization (similar to Python version)
    pub fn find_usable_range(&self, frequencies: &[f64], magnitudes: &[f64]) -> (usize, usize, f64, f64) {
        let min_db = -8.0;
        let avg_range = (200.0, 8000.0);
        let max_fails = if frequencies.len() >= 64 { 2 } else { 1 };

        // Calculate average level in the specified range
        let mut avg_sum = 0.0;
        let mut avg_count = 0;
        for i in 0..frequencies.len() {
            if frequencies[i] >= avg_range.0 && frequencies[i] <= avg_range.1 {
                avg_sum += magnitudes[i];
                avg_count += 1;
            }
        }
        
        if avg_count == 0 {
            // Fallback to full range if no frequencies in avg range
            return (0, frequencies.len() - 1, frequencies[0], frequencies[frequencies.len() - 1]);
        }

        let avg_db = avg_sum / avg_count as f64;
        
        // Find center frequency index (closest to 1kHz)
        let mut center_idx = 0;
        let mut min_diff = f64::INFINITY;
        for i in 0..frequencies.len() {
            let diff = (frequencies[i] - 1000.0).abs();
            if diff < min_diff {
                min_diff = diff;
                center_idx = i;
            }
        }

        // Find usable range going down from center
        let mut fails = 0;
        let mut low_idx = 0;
        for i in (0..center_idx).rev() {
            let normalized_mag = magnitudes[i] - avg_db;
            if normalized_mag < min_db {
                fails += 1;
            } else {
                fails = 0;
            }
            if fails >= max_fails {
                low_idx = (i + max_fails).min(frequencies.len() - 1);
                break;
            }
        }

        // Find usable range going up from center
        fails = 0;
        let mut high_idx = frequencies.len() - 1;
        for i in center_idx..frequencies.len() {
            let normalized_mag = magnitudes[i] - avg_db;
            if normalized_mag < min_db {
                fails += 1;
            } else {
                fails = 0;
            }
            if fails >= max_fails {
                high_idx = i.saturating_sub(max_fails);
                break;
            }
        }

        (low_idx, high_idx, frequencies[low_idx], frequencies[high_idx])
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

    /// Optimize EQ filters using a brute-force approach with weighted error calculation
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

        // Find usable frequency range for adaptive high-pass placement
        let (_low_idx, _high_idx, f_low, f_high) = self.find_usable_range(&frequencies, &measured_response);

        // Output usable frequency range information
        if self.output_progress {
            let measured_freqs: Vec<f64> = job.measured_curve.frequencies.clone();
            println!("Usable frequency range: {:.1} Hz - {:.1} Hz ({} frequency candidates)", 
                    f_low, f_high, self.generate_frequency_candidates(f_low, f_high, &measured_freqs).len());
        }

        // Generate target weights for the measured vs target comparison
        let target_weights = self.generate_target_weights(&job.target_curve, &frequencies, 
                                                        &measured_response, &target_response);

        // Calculate original error (weighted)
        let original_error = self.calculate_weighted_error(&measured_response, &target_response, &target_weights);
        
        let mut filters = Vec::new();
        let mut current_response = measured_response.clone();

        // Add adaptive high-pass filter if requested (maximum one allowed)
        if job.optimizer_params.add_highpass {
            // Calculate high-pass frequency as half of the lowest usable frequency (like Python)
            let hp_frequency = (f_low / 2.0).max(20.0).min(120.0); // Clamp between 20-120Hz for safety
            let hp_q = 0.5; // Use lower Q like Python for gentler slope
            
            let hp_filter = BiquadFilter::high_pass(hp_frequency, hp_q, self.sample_rate);
            let hp_response = hp_filter.frequency_response(&frequencies, self.sample_rate);
            
            // Apply high-pass response
            for i in 0..current_response.len() {
                current_response[i] += hp_response[i];
            }
            
            filters.push(hp_filter);
            
            // Recalculate weights for updated response
            let updated_weights = self.generate_target_weights(&job.target_curve, &frequencies, 
                                                             &current_response, &target_response);
            let error = self.calculate_weighted_error(&current_response, &target_response, &updated_weights);
            let step = OptimizationStep {
                step: 0,
                filters: filters.clone(),
                corrected_response: FrequencyResponse::new(frequencies.clone(), current_response.clone()),
                residual_error: error,
                message: format!("Added adaptive high-pass filter at {:.1}Hz (f_low={:.1}Hz)", hp_frequency, f_low),
                progress_percent: 0.0,
            };
            steps.push(step.clone());
            
            // Output step with optional frequency response if enabled
            if self.output_progress {
                if self.output_frequency_response {
                    let input_frequencies = &job.measured_curve.frequencies;
                    let original_magnitudes = &job.measured_curve.magnitudes_db;
                    let (magnitude_response, phase_response) = cascade_frequency_and_phase_response(&filters, input_frequencies, self.sample_rate);
                    self.output_step_with_frequency_response(&step, Some(input_frequencies), Some(&magnitude_response), Some(&phase_response), Some(original_magnitudes));
                } else {
                    self.output_step(&step);
                }
            }
        }

        // Optimization strategy: Brute force search for optimal PEQ filters
        // For each filter, try all combinations of frequency, Q, and gain
        // Keep the combination that provides the best improvement
        let mut used_frequencies: Vec<f64> = Vec::new();
        
        for filter_idx in 0..job.filter_count {
            let progress = (filter_idx as f64 / job.filter_count as f64) * 100.0;
            
            let mut best_filter: Option<BiquadFilter> = None;
            // Update weights for current response before calculating error
            let current_weights = self.generate_target_weights(&job.target_curve, &frequencies, 
                                                             &current_response, &target_response);
            let mut best_error = self.calculate_weighted_error(&current_response, &target_response, &current_weights);
            let mut best_response = current_response.clone();
            
            // Define search ranges - limit frequencies to usable range, add interpolated frequencies
            let measured_freqs: Vec<f64> = job.measured_curve.frequencies.clone();
            let freq_candidates: Vec<f64> = self.generate_frequency_candidates(f_low, f_high, &measured_freqs);
            let q_candidates = self.generate_q_candidates(&job.optimizer_params);
            let gain_candidates = self.generate_gain_candidates(&job.optimizer_params);
            
            // Try all combinations of frequency, Q, and gain
            for &freq in &freq_candidates {
                // Skip frequency if already used (with tolerance)
                let mut freq_already_used = false;
                for &used_freq in &used_frequencies {
                    if (freq / used_freq).abs().log2().abs() < 0.1 { // ~1/10 octave tolerance
                        freq_already_used = true;
                        break;
                    }
                }
                if freq_already_used {
                    continue;
                }
                
                for &q in &q_candidates {
                    for &gain_db in &gain_candidates {
                        // Create candidate filter
                        let candidate_filter = BiquadFilter::peaking_eq(freq, q, gain_db, self.sample_rate);
                        
                        // Calculate response with this filter added
                        let filter_response = candidate_filter.frequency_response(&frequencies, self.sample_rate);
                        let mut test_response = current_response.clone();
                        for i in 0..test_response.len() {
                            test_response[i] += filter_response[i];
                        }
                        
                        // Calculate weights for the test response and check error
                        let test_weights = self.generate_target_weights(&job.target_curve, &frequencies, 
                                                                      &test_response, &target_response);
                        let test_error = self.calculate_weighted_error(&test_response, &target_response, &test_weights);
                        
                        if test_error < best_error {
                            best_error = test_error;
                            best_filter = Some(candidate_filter);
                            best_response = test_response;
                        }
                    }
                }
            }
            
            // If we found an improvement, add the best filter
            if let Some(filter) = best_filter {
                // Track this frequency as used
                used_frequencies.push(filter.frequency);
                
                filters.push(filter.clone());
                current_response = best_response;
                
                let step = OptimizationStep {
                    step: filter_idx + 1,
                    filters: filters.clone(),
                    corrected_response: FrequencyResponse::new(frequencies.clone(), current_response.clone()),
                    residual_error: best_error,
                    message: format!("Added filter {} at {:.1}Hz, Q={:.1}, {:.1}dB", 
                                   filter_idx + 1, filter.frequency, filter.q, filter.gain_db),
                    progress_percent: progress,
                };
                steps.push(step.clone());
                
                // Output step with optional frequency response if progress is enabled
                if self.output_progress {
                    if self.output_frequency_response {
                        let input_frequencies = &job.measured_curve.frequencies;
                        let original_magnitudes = &job.measured_curve.magnitudes_db;
                        let (magnitude_response, phase_response) = cascade_frequency_and_phase_response(&filters, input_frequencies, self.sample_rate);
                        self.output_step_with_frequency_response(&step, Some(input_frequencies), Some(&magnitude_response), Some(&phase_response), Some(original_magnitudes));
                    } else {
                        self.output_step(&step);
                    }
                }
            } else {
                // No improvement found, stop optimization
                break;
            }
        }

        // Calculate final weighted error
        let final_weights = self.generate_target_weights(&job.target_curve, &frequencies, 
                                                       &current_response, &target_response);
        let final_error = self.calculate_weighted_error(&current_response, &target_response, &final_weights);
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
            usable_freq_low: f_low,
            usable_freq_high: f_high,
        }
    }
}
