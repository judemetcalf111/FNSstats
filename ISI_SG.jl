using Pkg
Pkg.activate(".")

# using DSP
using Random
using SavitzkyGolay
using StatsBase
using CSV
using DataFrames
using GLM 
using Statistics
using Plots
using ImageFiltering
# using Base.Threads
# using Foresight
using Distributions
# Foresight.set_theme!(foresight(:physics))

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI_SG"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

function get_sg_velocity(x, y, window_size, dt)

    # Ensure odd:
    if window_size % 2 == 0
        window_size -= 1
    end
    
    # order: The polynomial order. 
    # 2 (quadratic) or 3 (cubic) is standard. 
    # 3 is better at keeping sharp peaks, 2 is smoother.
    poly_order = 2
    
    # deriv=1 calculates velocity directly.
    # rate=1/dt automatically scales the output to units/second.
    # The function returns a struct, the data is in the .y field
    sg_x = savitzky_golay(x, window_size, poly_order, deriv=1, rate=1/dt)
    sg_y = savitzky_golay(y, window_size, poly_order, deriv=1, rate=1/dt)
    
    # Combine into velocity magnitude
    # We access the .y field to get the vector data
    vx = sg_x.y
    vy = sg_y.y
    
    v = sqrt.(vx.^2 .+ vy.^2)
    return v
end

function calculate_isi(x, y; timeout_smoother=0.007, timeout=0.150, λ=6.0, ratio=0.1, dt=0.0001, min_dur_steps=5)
    timeout_steps = Int(ceil(timeout/dt))
    timeout_smoother_steps = Int(ceil(timeout_smoother/dt))
    
    v = get_sg_velocity(x, y, timeout_smoother_steps, 0.0001)

    # Threshold: λ=6 is standard for Engbert-Kliegl. 
    # If your data is very clean, you can lower this to 4 or 5.
    msd = median(abs.(v .- median(v))) / 0.6745
    threshold_on = λ * msd

    λ_off = ratio * λ
    threshold_off = λ_off * msd

    starts_timed = Int[]
    last_start = -timeout_steps 
    
    i = 1
    n = length(v)
    
    while i <= n
        # 1. Wait for velocity to spike above the high onset threshold
        if v[i] > threshold_on
            j = i + 1
            
            # 2. Keep moving forward until the particle is "effectively stationary"
            while j <= n && v[j] >= threshold_off
                j += 1
            end
            
            event_duration = j - i
            
            # 3. Check duration and Refractory period
            if event_duration >= min_dur_steps && (i - last_start) > timeout_steps
                push!(starts_timed, i)
                last_start = i
            end
            
            # 4. Advance our search index to the end of the stationary tail
            i = j
        else
            i += 1
        end
    end

    isi = diff(starts_timed) .* (dt * 1000) 
    return (starts_timed, isi) 
end

# loop through csv files in /datadir
for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.0001

    # 3. CRITICAL PARAMETER FIX:
    # timeout_smoother: 0.005 (5ms) instead of 0.2 (200ms)
    # timeout: 0.2 (200ms refractory) instead of 10 (10 seconds)
    (sac_starts, isi) = calculate_isi(x, y; 
        timeout_smoother=30, 
        timeout=0.2, 
        ratio=0.1,
        λ=6.0, 
        dt=dt, 
        min_dur_steps=500
    )

    if length(isi) > 5 # Only plot if we have decent statistics
        # timelength over which we plot:
        timelength = 50000.
        # Precomputing the log normal fit
        lISI = log.(isi)
        log_σ = std(lISI)
        log_peak = median(lISI)
        plotting_x = range(0,timelength)
        fitted_ln = LogNormal(log_peak,log_σ)

        println("WE GET A LOG SD OF:        ", round(log_σ, digits=3))
        
        p = histogram(isi,
                    bins=range(0, timelength, length=40),
                    normalize=:pdf,
                    xlabel="Inter-Saccadic Interval (ms)",
                    # title="File: $(basename(file))",
                    xlims=(0, timelength),
                    legend=false)

        plot!(plotting_x,pdf.(fitted_ln,plotting_x))

        outname = joinpath(output_dir, splitext(basename(file))[1] * "-ISI_TRANS.pdf")
        savefig(p, outname)

        lags = 0:length(isi)-1
        acf_values = autocor(isi, lags)

        p2 = plot(lags, acf_values, xlims=(0,40), ylims=(-0.1,0.2), xlabel="Lag", ylabel="Autocorrelation", title="Autocorrelation of ISIs")
        outname2 = joinpath(output_dir, splitext(basename(file))[1] * "-ISI_ACF.pdf")

        shuffled_acf = autocor(shuffle(isi), lags)
        plot!(lags, shuffled_acf, label="Shuffled ISIs", color=:red, linestyle=:dash)
        savefig(p2, outname2)

        p3 = plot(isi, xlabel="Index", ylabel="ISI (ms)", title="ISI timeseries")
        outname3 = joinpath(output_dir, splitext(basename(file))[1] * "-ISI_TIMESERIES.pdf")
        savefig(p3, outname3)

        p4 = plot(x, label="X Position", color=:blue, alpha=0.6)
        plot!(y, label="Y Position", color=:green, alpha=0.6)
        
        # Add scatter points on X trace to mark starts
        plot!(sac_starts, x[sac_starts], seriestype=:scatter, 
              markersize=3, color=:red, label="Detected Saccades")
        
        plot!(xlims=(100000, 200000), ylims=(-15,15), title="Saccade Detection Trace")
        
        outname4 = joinpath(output_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
        savefig(p4, outname4)
    end
    
    println("Processed $FigNumber: $(basename(file))")
    global FigNumber += 1
end
