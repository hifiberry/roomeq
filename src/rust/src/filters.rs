use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use num_complex::Complex;

/// Represents a weight parameter for target curves
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Weight {
    Single(f64),
    Tuple(f64, f64),
    Triple(f64, f64, f64),
}

/// Represents a control point in a target curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePoint {
    pub frequency: f64,
    pub target_db: f64,
    pub weight: Option<Weight>,
}

/// Target curve definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCurve {
    pub name: String,
    pub description: String,
    pub expert: bool,
    pub curve: Vec<CurvePoint>,
}

/// Optimizer preset parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerPreset {
    pub name: String,
    pub description: String,
    pub qmax: f64,
    pub mindb: f64,
    pub maxdb: f64,
    pub add_highpass: bool,
}

/// Frequency response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyResponse {
    pub frequencies: Vec<f64>,
    pub magnitudes_db: Vec<f64>,
}

impl FrequencyResponse {
    pub fn new(frequencies: Vec<f64>, magnitudes_db: Vec<f64>) -> Self {
        assert_eq!(frequencies.len(), magnitudes_db.len(), "Frequencies and magnitudes must have same length");
        Self { frequencies, magnitudes_db }
    }

    pub fn len(&self) -> usize {
        self.frequencies.len()
    }

    pub fn interpolate_at(&self, frequency: f64) -> f64 {
        if frequency <= self.frequencies[0] {
            return self.magnitudes_db[0];
        }
        if frequency >= self.frequencies[self.frequencies.len() - 1] {
            return self.magnitudes_db[self.magnitudes_db.len() - 1];
        }

        // Find the two points to interpolate between
        let mut i = 0;
        while i < self.frequencies.len() - 1 && self.frequencies[i + 1] < frequency {
            i += 1;
        }

        // Ensure i is within valid range for i+1 access
        if i >= self.frequencies.len() - 1 {
            return self.magnitudes_db[self.magnitudes_db.len() - 1];
        }

        let f1 = self.frequencies[i];
        let f2 = self.frequencies[i + 1];
        let m1 = self.magnitudes_db[i];
        let m2 = self.magnitudes_db[i + 1];

        // Linear interpolation in log-frequency space
        let log_f = frequency.ln();
        let log_f1 = f1.ln();
        let log_f2 = f2.ln();
        
        let t = (log_f - log_f1) / (log_f2 - log_f1);
        m1 + t * (m2 - m1)
    }
}

/// Biquad filter coefficients and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiquadFilter {
    pub filter_type: String,
    pub frequency: f64,
    pub q: f64,
    pub gain_db: f64,
    pub description: String,
    pub coefficients: BiquadCoefficients,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiquadCoefficients {
    pub b: [f64; 3], // [b0, b1, b2]
    pub a: [f64; 3], // [a0, a1, a2]
}

impl BiquadFilter {
    pub fn high_pass(frequency: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            filter_type: "hp".to_string(),
            frequency,
            q,
            gain_db: 0.0,
            description: format!("High pass {:.1}Hz", frequency),
            coefficients: BiquadCoefficients {
                b: [b0, b1, b2],
                a: [a0, a1, a2],
            }
        }
    }

    pub fn peaking_eq(frequency: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let a_factor = 10.0_f64.powf(gain_db / 40.0);
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a_factor;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a_factor;
        let a0 = 1.0 + alpha / a_factor;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a_factor;

        Self {
            filter_type: "eq".to_string(),
            frequency,
            q,
            gain_db,
            description: format!("Peaking EQ {:.1}Hz {:.1}dB", frequency, gain_db),
            coefficients: BiquadCoefficients {
                b: [b0, b1, b2],
                a: [a0, a1, a2],
            }
        }
    }

    /// Calculate frequency response of this filter at given frequencies
    pub fn frequency_response(&self, frequencies: &[f64], sample_rate: f64) -> Vec<f64> {
        frequencies.iter().map(|&freq| {
            let w = 2.0 * PI * freq / sample_rate;
            let z = Complex::new(0.0, w).exp();

            let numerator = self.coefficients.b[0] + 
                           self.coefficients.b[1] * z.inv() + 
                           self.coefficients.b[2] * z.inv() * z.inv();
            
            let denominator = self.coefficients.a[0] + 
                             self.coefficients.a[1] * z.inv() + 
                             self.coefficients.a[2] * z.inv() * z.inv();

            let h = numerator / denominator;
            20.0 * h.norm().log10()
        }).collect()
    }

    pub fn as_text(&self) -> String {
        match self.filter_type.as_str() {
            "hp" => format!("hp:{:.1}:{:.3}", self.frequency, self.q),
            "eq" => format!("eq:{:.1}:{:.3}:{:.2}", self.frequency, self.q, self.gain_db),
            _ => format!("coeff:{:.6}:{:.6}:{:.6}:{:.6}:{:.6}:{:.6}", 
                        self.coefficients.a[0], self.coefficients.a[1], self.coefficients.a[2],
                        self.coefficients.b[0], self.coefficients.b[1], self.coefficients.b[2]),
        }
    }
}

/// Calculate cascaded frequency response of multiple filters
#[allow(dead_code)]
pub fn cascade_frequency_response(filters: &[BiquadFilter], frequencies: &[f64], sample_rate: f64) -> Vec<f64> {
    if filters.is_empty() {
        return vec![0.0; frequencies.len()];
    }

    let mut total_response = vec![0.0; frequencies.len()];
    
    for filter in filters {
        let filter_response = filter.frequency_response(frequencies, sample_rate);
        for (i, response) in filter_response.iter().enumerate() {
            total_response[i] += response;
        }
    }

    total_response
}
