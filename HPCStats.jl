using Distributed
using ClusterManagers # Optional: Use if submitting via SLURM/PBS
using Pkg

# Add workers
if length(workers()) == 1
    addprocs(Sys.CPU_THREADS - 1) 
end

@everywhere using CSV, DataFrames, Statistics, FFTW, StatsBase

# --- Define kernels (Available to all workers) ---
@everywhere begin
    # Optimized MAD function
    function calc_metrics_single_file(filepath, max_lags)
        # Fast read - avoiding full DataFrame features if possible for speed
        df = CSV.read(filepath, DataFrame)
        x = df.value1
        y = df.value2
        
        N = length(x)
        
        # --- MAD Calculation ---
        # User definition: (Mean(|dx|) + Mean(|dy|))^2
        # We calculate the sum here, squaring is non-linear so we do it per file
        mad_curve = zeros(Float64, max_lags)
        
        for tau in 1:max_lags
            # Using views prevents memory allocation for slices
            vx_head = view(x, 1+tau:N)
            vx_tail = view(x, 1:N-tau)
            vy_head = view(y, 1+tau:N)
            vy_tail = view(y, 1:N-tau)
            
            # Mean Absolute Deviation for this lag
            # Note: We implement the math exactly as you had it
            mean_abs_dx = mean(abs.(vx_head .- vx_tail))
            mean_abs_dy = mean(abs.(vy_head .- vy_tail))
            
            mad_curve[tau] = (mean_abs_dx + mean_abs_dy)^2
        end

        # --- ACF Calculation ---
        # Calculate lags up to max_lags
        acf_x = autocor(x, 1:max_lags)
        acf_y = autocor(y, 1:max_lags)
        
        # --- FFT Calculation ---
        # Using steps
        Δx = [0;diff(x)]
        Δy = [0;diff(y)]

        Δs = sqrt.(Δx.^2 + Δy.^2)

        PSD = abs.(rfft(Δs)).^2
        
        # Return a tuple of vectors
        return (mad_curve, acf_x, acf_y, PSD)
    end
end

# --- 3. Main Execution Block ---
function main()
    input_dir  = "/suphys/jmet9477/export/FulldFNS/data/Long/" # Update for HPC path
    output_dir = "/suphys/jmet9477/export/FulldFNS/data/Longanal/"
    isdir(output_dir) || mkdir(output_dir)
    
    files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    n_files = length(files)
    
    println("Found $n_files files. Starting parallel processing...")
    
    # PARAMETERS
    MAX_LAGS = 20000 
    
    # --- PARALLEL REDUCTION ---
    # @distributed (+) means: execute the loop in chunks on workers, 
    # and sum (+) the results of the tuple element-wise.
    (sum_mad, sum_acf_x, sum_acf_y, PSD_sum) = @distributed (+) for f in files
        calc_metrics_single_file(f, MAX_LAGS)
    end
    
    # --- AVERAGING ---
    # Divide sums by N first to get the Ensemble Average
    avg_mad   = sum_mad ./ n_files
    avg_acf_x = sum_acf_x ./ n_files
    avg_acf_y = sum_acf_y ./ n_files
    avg_PSD = PSD_sum ./ n_files

    println("Processing complete. Saving aggregate data...")

    # --- SAVE RESULTS ---
    # Save processed data to CSV so you can plot later without re-running
    df_results = DataFrame(
        Lag = 1:MAX_LAGS,
        MAD = avg_mad,
        ACF_X = avg_acf_x,
        ACF_Y = avg_acf_y
    )
    
    # Saving FFT separately
    df_fft = DataFrame(
        FreqIndex = 1:length(avg_PSD),
        PSD = avg_PSD,
    )
    
    CSV.write(joinpath(output_dir, "Ensemble_TimeDomain.csv"), df_results)
    CSV.write(joinpath(output_dir, "Ensemble_FreqDomain.csv"), df_fft)
    
    println("Done.")
end

main()
