mod filters;
mod optimizer;

use optimizer::*;

use serde_json;
use std::io::{self, Read};
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();
    
    info!("RoomEQ Optimizer v0.6.0 starting...");
    
    // Read optimization job from stdin
    let mut stdin = io::stdin();
    let mut input = String::new();
    stdin.read_to_string(&mut input)?;

    // Parse optimization job
    let job: OptimizationJob = match serde_json::from_str(&input.trim()) {
        Ok(job) => job,
        Err(e) => {
            eprintln!("Error parsing input JSON: {}", e);
            std::process::exit(1);
        }
    };

    info!("Optimization job received:");
    info!("  Measured curve: {} points", job.measured_curve.len());
    info!("  Target curve: {}", job.target_curve.name);
    info!("  Optimizer: {}", job.optimizer_params.name);
    info!("  Filter count: {}", job.filter_count);
    info!("  Sample rate: {}Hz", job.sample_rate);

    // Create optimizer
    let optimizer = RoomEQOptimizer::new(job.sample_rate);

    // Run optimization (this will output steps to stdout during processing)
    let result = optimizer.optimize(job);

    // Output final result to stderr so it doesn't interfere with step output
    eprintln!("Optimization completed:");
    eprintln!("  Success: {}", result.success);
    eprintln!("  Filters created: {}", result.filters.len());
    eprintln!("  Original error: {:.2} dB RMS", result.original_error);
    eprintln!("  Final error: {:.2} dB RMS", result.final_error);
    eprintln!("  Improvement: {:.1} dB", result.improvement_db);
    eprintln!("  Processing time: {} ms", result.processing_time_ms);

    if let Some(ref error_msg) = result.error_message {
        eprintln!("  Error: {}", error_msg);
        std::process::exit(1);
    }

    info!("Optimization complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{BiquadFilter, FrequencyResponse, cascade_frequency_response};

    #[test]
    fn test_biquad_filter_creation() {
        let hp = BiquadFilter::high_pass(80.0, 0.707, 48000.0);
        assert_eq!(hp.filter_type, "hp");
        assert_eq!(hp.frequency, 80.0);
        assert!((hp.q - 0.707).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_response_interpolation() {
        let frequencies = vec![100.0, 1000.0, 10000.0];
        let magnitudes = vec![-3.0, 0.0, -6.0];
        let response = FrequencyResponse::new(frequencies, magnitudes);

        // Test interpolation at 500Hz (between 100 and 1000)
        let result = response.interpolate_at(500.0);
        assert!(result > -3.0 && result < 0.0);
    }

    #[test]
    fn test_cascade_response() {
        let filters = vec![
            BiquadFilter::peaking_eq(1000.0, 2.0, 3.0, 48000.0),
            BiquadFilter::high_pass(80.0, 0.707, 48000.0),
        ];
        
        let frequencies = vec![100.0, 1000.0, 10000.0];
        let response = cascade_frequency_response(&filters, &frequencies, 48000.0);
        
        assert_eq!(response.len(), frequencies.len());
        // At 1kHz, should have significant boost from the peaking filter
        assert!(response[1] > 1.0);
    }

    #[test]
    fn test_error_calculation() {
        let optimizer = RoomEQOptimizer::new(48000.0);
        let measured = vec![0.0, 1.0, 2.0];
        let target = vec![1.0, 1.0, 1.0];
        
        let error = optimizer.calculate_error(&measured, &target);
        let expected = ((1.0 + 0.0 + 1.0) / 3.0_f64).sqrt();
        assert!((error - expected).abs() < 1e-6);
    }
}
