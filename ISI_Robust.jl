using Pkg
Pkg.activate(".")

# using DSP
using Random
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
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI_DETECT"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

function calculate_isi(x, y; timeout_smoother=0.007, timeout=0.150, λ=8.0, dt=0.001, min_dur_steps=5)
    timeout_steps = Int(ceil(timeout/dt))
    timeout_smoother_steps = Int(ceil(timeout_smoother/dt))
    
    # 2. FIXED: Explicit kernel construction for clarity
    # A sigma of 7ms is enough to kill white noise but keeps the velocity peak
    function gaussian_smooth(vec, σ)
        return imfilter(vec, Kernel.gaussian((σ,)))
    end

    x_sm = gaussian_smooth(x, timeout_smoother_steps) 
    y_sm = gaussian_smooth(y, timeout_smoother_steps)
    
    vx = [0.0; diff(x_sm)] ./ dt
    vy = [0.0; diff(y_sm)] ./ dt
    v = sqrt.(vx.^2 .+ vy.^2)

    # Threshold: λ=6 is standard for Engbert-Kliegl. 
    # If your data is very clean, you can lower this to 4 or 5.
    msd = median(abs.(v .- median(v))) / 0.6745
    threshold = λ * msd
    is_saccade = v .> threshold

    starts_timed = Int[]
    last_start = -timeout_steps 
    
    i = 1
    while i < length(is_saccade)
        if is_saccade[i]
            j = i
            while j <= length(is_saccade) && is_saccade[j]
                j += 1
            end
            event_duration = j - i
            
            # Check duration and Refractory period
            if event_duration >= min_dur_steps && (i - last_start) > timeout_steps
                push!(starts_timed, i)
                last_start = i
            end
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
    dt = 0.001

    (sac_starts, isi) = calculate_isi(x, y; 
        timeout_smoother=20,
        timeout=0.3,
        λ=3, 
        dt=dt, 
        min_dur_steps=100
    )

    if length(isi) > 5 # Only plot if we have decent statistics
        # timelength over which we plot:
        timelength = 25000.
        # Lognormal fit
        lISI = log.(isi)
        log_σ = std(lISI)
        log_peak = median(lISI)
        plotting_x = range(0,timelength)
        fitted_ln = LogNormal(log_peak,log_σ)

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
        
        plot!(xlims=(10000, 300000), title="Saccade Detection Trace")
        
        outname4 = joinpath(output_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
        savefig(p4, outname4)

        ### Poincaré Plot ###

        xs = isi[1:end-1]
        ys = isi[2:end]

        diff_vec = ys .- xs       # Perpendicular changes
        sum_vec = ys .+ xs        # Longitudinal changes

        sd1 = std(diff_vec) / sqrt(2)
        sd2 = std(sum_vec) / sqrt(2)

        println("Rhythm Stability Metrics:")
        println("SD1 (Short-term jitter): ", round(sd1, digits=2))
        println("SD2 (Long-term drift):   ", round(sd2, digits=2))
        println("Ratio (SD2/SD1):         ", round(sd2/sd1, digits=2))
        println("Log SD:                  ", round(log_σ, digits=3))

        poinc_plot = scatter(xs, ys, 
            label="Intervals", 
            alpha=0.6, 
            markerstrokewidth=0,
            color=:blue,
            aspect_ratio=:equal, # Important to see true shape
            title="Poincaré Plot (Return Map)",
            xlabel="Interval t (ms)",
            ylabel="Interval t+1 (ms)"
        )

        # Add Line of Identity (Perfect Rhythm Line)
        plot!([minimum(isi), maximum(isi)], [minimum(isi), maximum(isi)], 
            label="Line of Identity (y=x)", color=:red, linestyle=:dash)

        poinc_outname = joinpath(output_dir, splitext(basename(file))[1] * "-POINCARE.pdf")
        savefig(poinc_plot,poinc_outname)
    end
    
    println("Processed $FigNumber: $(basename(file))")
    global FigNumber += 1
end
