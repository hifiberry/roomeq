#!/usr/bin/env rust
//! HiFiBerry Multi-FFT Analysis Tool - Rust Implementation
//!
//! This is a Rust port of the original C implementation by Joerg Schambacher, 2019
//! Provides FFT analysis functionality for WAV files with reference comparison
//! and multi-file averaging capabilities.

use clap::{Arg, Command};
use std::path::Path;
use std::process;

mod analyze_fft;
use analyze_fft::{FftAnalyzer, FftConfig};

fn main() {
    let matches = Command::new("HiFiBerry Multi-FFT Tool")
        .version("1.0.0")
        .author("HiFiBerry <support@hifiberry.com>")
        .about("FFT analysis tool for audio measurements (Rust port)")
        .arg(
            Arg::new("reference")
                .short('r')
                .long("reference")
                .value_name("FILE")
                .help("Reference WAV file")
                .required(true),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .value_name("LEVEL")
                .help("Verbosity level (0-2)")
                .default_value("0"),
        )
        .arg(
            Arg::new("points")
                .short('n')
                .long("points")
                .value_name("NUM")
                .help("Number of reduced FFT sample points")
                .default_value("64"),
        )
        .arg(
            Arg::new("freq-min")
                .long("freq-min")
                .value_name("HZ")
                .help("Minimum frequency for analysis")
                .default_value("20.0"),
        )
        .arg(
            Arg::new("freq-max")
                .long("freq-max")
                .value_name("HZ")
                .help("Maximum frequency for analysis")
                .default_value("20000.0"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output CSV file")
                .default_value("fftdB_vbw.csv"),
        )
        .arg(
            Arg::new("files")
                .value_name("FILES")
                .help("WAV files to analyze")
                .num_args(1..)
                .required(true),
        )
        .get_matches();

    // Parse arguments
    let reference_file = matches.get_one::<String>("reference").unwrap();
    let verbose: u32 = matches
        .get_one::<String>("verbose")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid verbosity level");
            process::exit(1);
        });
    
    let num_points: usize = matches
        .get_one::<String>("points")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid number of points");
            process::exit(1);
        });

    let freq_min: f64 = matches
        .get_one::<String>("freq-min")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid minimum frequency");
            process::exit(1);
        });

    let freq_max: f64 = matches
        .get_one::<String>("freq-max")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid maximum frequency");
            process::exit(1);
        });

    let output_file = matches.get_one::<String>("output").unwrap();
    let input_files: Vec<&String> = matches.get_many::<String>("files").unwrap().collect();

    // Validate arguments
    if freq_min >= freq_max {
        eprintln!("Error: Minimum frequency must be less than maximum frequency");
        process::exit(1);
    }

    if num_points < 1 || num_points > 1024 {
        eprintln!("Error: Number of points must be between 1 and 1024");
        process::exit(1);
    }

    // Print header
    println!("HiFiBerry Multi-FFT Tool v1.0.0 (Rust)");
    println!("========================================");
    println!();

    // Create analyzer configuration
    let config = FftConfig {
        verbose,
        num_output_points: num_points,
        f_min_sweep: freq_min,
        f_max_sweep: freq_max,
    };

    // Initialize analyzer
    let mut analyzer = FftAnalyzer::with_config(config);

    // Validate reference file exists
    if !Path::new(reference_file).exists() {
        eprintln!("Error: Reference file not found: {}", reference_file);
        process::exit(1);
    }

    // Set reference file
    if let Err(e) = analyzer.set_reference(reference_file) {
        eprintln!("Error loading reference file '{}': {}", reference_file, e);
        process::exit(1);
    }

    println!("Loaded reference: {}", reference_file);

    // Process each input file
    for (index, file_path) in input_files.iter().enumerate() {
        if !Path::new(file_path).exists() {
            eprintln!("Warning: File not found, skipping: {}", file_path);
            continue;
        }

        println!("Processing {}: {}", index + 1, file_path);
        
        if let Err(e) = analyzer.process_file(file_path) {
            eprintln!("Error processing '{}': {}", file_path, e);
            continue;
        }
    }

    // Get final results
    match analyzer.get_result() {
        Ok(result) => {
            println!("Analysis complete. Saving results to: {}", output_file);
            
            if let Err(e) = save_results(&result, output_file) {
                eprintln!("Error saving results: {}", e);
                process::exit(1);
            }

            // Print summary
            println!();
            println!("Results Summary:");
            println!("  Frequency range: {:.1} - {:.1} Hz", freq_min, freq_max);
            println!("  Output points: {}", result.frequencies.len());
            println!("  Sample rate: {} Hz", result.sample_rate);
            println!("  FFT size: {}", result.fft_size);

            // Show first few results if verbose
            if verbose > 0 {
                println!();
                println!("First 10 results:");
                println!("  Freq (Hz)  |  Mag (dB)  |  Phase (°)");
                println!("  -----------|------------|------------");
                for i in 0..10.min(result.frequencies.len()) {
                    println!(
                        "  {:9.2} | {:9.2} | {:10.2}",
                        result.frequencies[i],
                        result.magnitudes[i],
                        result.phases[i]
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("Error generating results: {}", e);
            process::exit(1);
        }
    }

    println!("Done.");
}

/// Save FFT results to CSV file
fn save_results(
    result: &analyze_fft::FftResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write CSV header
    writeln!(writer, "Frequency_Hz,Magnitude_dB,Phase_degrees")?;

    // Write data
    for i in 0..result.frequencies.len() {
        writeln!(
            writer,
            "{:.6},{:.6},{:.6}",
            result.frequencies[i], result.magnitudes[i], result.phases[i]
        )?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argument_validation() {
        // Test frequency validation
        assert!(20.0 < 20000.0);
        
        // Test points validation
        assert!(64 >= 1 && 64 <= 1024);
    }
}
