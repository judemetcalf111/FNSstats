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
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI_KMEAN"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

function detect_transitions_geometric(x, y; n_clusters=3, min_dwell=10000)
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
    
    current_length = 0

    for i in 2:lastindex(assignments)
        if assignments[i] != current_state
            # Potential switch. 
            # Check if we have stayed in the NEW state long enough to call it real?
            # Or (simpler for retrospective): Did we stay in the OLD state long enough?
            
            duration_in_previous = i - last_switch_time

            if assignments[i] == assignments[i-1]
                current_length += 1
            else
                current_length = 0
            end
            
            if duration_in_previous > min_dwell && current_length >= 30
                push!(transitions, i) # This index is where the transition started
                current_state = assignments[i]
                last_switch_time = i
                current_length = 0
            else
                # If the duration was too short, it was just boundary noise.
                # We essentially ignore the flicker.
            end
        end
    end
    
    return transitions, R.centers
end

isi_vec = []

# loop through csv files in /datadir
for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.001

    (sac_starts, centres) = detect_transitions_geometric(x,y; n_clusters=3, min_dwell=1000)

    isi = diff(sac_starts)

    append!(isi_vec,isi)

    if length(isi) > 5 # Only plot if we have decent statistics
        # lags = 0:length(isi)-1
        # acf_values = autocor(isi, lags)
        
        # p2 = plot(lags, acf_values, xlims=(0,40), ylims=(-0.1,0.2), xlabel="Lag", ylabel="Autocorrelation", title="Autocorrelation of ISIs")
        # outname2 = joinpath(output_dir, splitext(basename(file))[1] * "-ISI_ACF.pdf")
        
        # shuffled_acf = autocor(shuffle(isi), lags)
        # plot!(lags, shuffled_acf, label="Shuffled ISIs", color=:red, linestyle=:dash)
        # savefig(p2, outname2)
        
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

# timelength over which we plot:
timelength = 120000.
# Precomputing the log normal fit
lISI = log.(isi_vec)
log_σ = std(lISI)
log_peak = median(lISI)
plotting_x = range(0,timelength)
fitted_ln = LogNormal(log_peak,log_σ)

p = histogram(isi_vec,
bins=range(0, timelength, length=60),
normalize=:pdf,
xlabel="Inter-Saccadic Interval (ms)",
# title="File: $(basename(file))",
xlims=(0, timelength),
legend=false)

plot!(plotting_x,pdf.(fitted_ln,plotting_x))

outname = joinpath(output_dir, "Total_ISIs.pdf")
savefig(p, outname)

p3 = plot(isi_vec, xlabel="Index", ylabel="ISI (ms)", title="ISI timeseries")
outname3 = joinpath(output_dir, "ISI_TIMESERIES.pdf")
savefig(p3, outname3)

### Poincaré Plot ###

xs = isi_vec[1:end-1]
ys = isi_vec[2:end]

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
plot!([minimum(isi_vec), maximum(isi_vec)], [minimum(isi_vec), maximum(isi_vec)], 
label="Line of Identity (y=x)", color=:red, linestyle=:dash)

poinc_outname = joinpath(output_dir, "POINCARE.pdf")
savefig(poinc_plot,poinc_outname)
