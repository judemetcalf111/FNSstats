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
using Clustering
using Distances
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

function detect_transitions_geometric(x, y; n_clusters=3, min_dwell=50000)
    # 1. Combine data
    data = hcat(x, y)' # Clustering expects (n_features, n_samples)
    
    # 2. Find the 3 centers (attractors) automatically
    # This assumes the particle spends most time in the wells, not the transition paths
    R = kmeans(data, n_clusters)
    assignments = R.assignments # distinct integers (1, 2, 3) for each timepoint
    
    # 3. Detect switches
    # A switch happens when assignment[i] != assignment[i-1]
    # However, at the boundary, noise might cause 1-2-1-2 flickering.
    # We apply a "Minimum Dwell Time" (Hysteresis)
    
    transitions = Int[]
    current_state = assignments[1]
    last_switch_time = 1

    for i in 2:lastindex(assignments)
        if assignments[i] != current_state
            # Potential switch. 
            # Check if we have stayed in the NEW state long enough to call it real?
            # Or (simpler for retrospective): Did we stay in the OLD state long enough?
            
            duration_in_previous = i - last_switch_time
            
            if duration_in_previous > min_dwell
                push!(transitions, i) # This index is where the transition started
                current_state = assignments[i]
                last_switch_time = i
            else
                # If the duration was too short, it was just boundary noise.
                # We essentially ignore the flicker.
            end
        end
    end
    
    return transitions, R.centers
end

# loop through csv files in /datadir
for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.001

    # 3. CRITICAL PARAMETER FIX:
    # timeout_smoother: 0.005 (5ms) instead of 0.2 (200ms)
    # timeout: 0.2 (200ms refractory) instead of 10 (10 seconds)
    # (sac_starts, isi) = calculate_isi(x, y; 
    #     timeout_smoother=0.1, 
    #     timeout=0.2, 
    #     λ=10.0, 
    #     dt=dt, 
    #     min_dur_steps=100
    # )

    (sac_starts, centres) = detect_transitions_geometric(x,y)

    isi = diff(sac_starts)

    if length(isi) > 5 # Only plot if we have decent statistics
        # timelength over which we plot:
        timelength = 25000.
        # Precomputing the log normal fit
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
    end
    
    println("Processed $FigNumber: $(basename(file))")
    global FigNumber += 1
end
