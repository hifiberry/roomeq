mod filters;
mod optimizer;

use optimizer::*;

use serde_json;
use std::io::{self, Read};
use log::info;
use clap::Parser;

/// High-performance RoomEQ optimizer for audio equalization
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output progress steps during optimization
    #[arg(long)]
    progress: bool,

    /// Output final result after optimization
    #[arg(long)]
    result: bool,

    /// Output human-readable text instead of JSON
    #[arg(long)]
    human_readable: bool,

    /// Output frequency response after each filter step
    #[arg(long)]
    frequency_response: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize logger only if human-readable output is requested
    // Otherwise it would interfere with JSON output
    if args.human_readable {
        env_logger::init();
        info!("RoomEQ Optimizer v0.6.0 starting...");
    }
    
    // Read optimization job from stdin
    let mut stdin = io::stdin();
    let mut input = String::new();
    stdin.read_to_string(&mut input)?;

    // Parse optimization job
    let job: OptimizationJob = match serde_json::from_str(&input.trim()) {
        Ok(job) => job,
        Err(e) => {
            if args.human_readable {
                eprintln!("Error parsing input JSON: {}", e);
            } else {
                let error_json = serde_json::json!({
                    "type": "error",
                    "message": format!("Error parsing input JSON: {}", e)
                });
                println!("{}", error_json);
            }
            std::process::exit(1);
        }
    };

    if args.human_readable {
        info!("Optimization job received:");
        info!("  Measured curve: {} points", job.measured_curve.len());
        info!("  Target curve: {}", job.target_curve.name);
        info!("  Optimizer: {}", job.optimizer_params.name);
        info!("  Filter count: {}", job.filter_count);
        info!("  Sample rate: {}Hz", job.sample_rate);
    }

    // Create optimizer with output mode configuration
    let mut optimizer = RoomEQOptimizer::new(job.sample_rate);
    optimizer.set_output_mode(args.progress, args.human_readable, args.frequency_response);

    // Run optimization
    let result = optimizer.optimize(job);

    // Output final result if requested
    if args.result {
        if args.human_readable {
            println!("\n=== OPTIMIZATION RESULTS ===");
            println!("Success: {}", result.success);
            println!("Filters created: {}", result.filters.len());
            println!("Usable frequency range: {:.1} Hz - {:.1} Hz", result.usable_freq_low, result.usable_freq_high);
            println!("Original error: {:.2} dB RMS", result.original_error);
            println!("Final error: {:.2} dB RMS", result.final_error);
            println!("Improvement: {:.1} dB", result.improvement_db);
            println!("Processing time: {} ms", result.processing_time_ms);
            
            if !result.filters.is_empty() {
                println!("\nFilters:");
                for (i, filter) in result.filters.iter().enumerate() {
                    println!("  {}: {}", i + 1, filter.as_text());
                }
            }

            if let Some(ref error_msg) = result.error_message {
                println!("Error: {}", error_msg);
                std::process::exit(1);
            }
        } else {
            // Output JSON result
            let result_json = serde_json::to_string(&result)?;
            println!("{}", result_json);
        }
    }

    // Exit with error code if optimization failed
    if !result.success {
        std::process::exit(1);
    }

    if args.human_readable {
        info!("Optimization complete");
    }
    
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
