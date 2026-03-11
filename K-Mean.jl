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

# 1. Pool a subset of data from all files to find the 9 global centers
println("Phase 1: Finding 9 global centers...")
all_x = Float64[]
all_y = Float64[]

for file in csv_files
    df = CSV.read(file, DataFrame)
    # Take every 10th point to save memory, plenty for finding stable centers
    append!(all_x, df.value1[1:10:end]) 
    append!(all_y, df.value2[1:10:end])
end

global_data = hcat(all_x, all_y)'
R_global = kmeans(global_data, 9) # k=9 for the 9-dot grid
global_centers = R_global.centers
println("Global centers established!")

FigNumber = 1
# Function to assign every x,y coordinate to the nearest global center
function assign_to_centers(x, y, centers)
    n = length(x)
    k_clusters = size(centers, 2)
    assignments = zeros(Int, n)
    
    for i in 1:n
        min_dist = Inf
        best_c = 1
        for c in 1:k_clusters
            # Calculate squared distance to each center
            dist = (x[i] - centers[1, c])^2 + (y[i] - centers[2, c])^2
            if dist < min_dist
                min_dist = dist
                best_c = c
            end
        end
        assignments[i] = best_c
    end
    return assignments
end

# Fixed Hysteresis logic
function detect_transitions(assignments; min_dwell=1000, min_stable=30)
    transitions = Int[]
    
    committed_state = assignments[1]
    time_in_committed = 1
    
    current_streak_state = assignments[1]
    current_streak_len = 1
    
    for i in 2:length(assignments)
        # 1. Track the current continuous streak of any state
        if assignments[i] == current_streak_state
            current_streak_len += 1
        else
            current_streak_state = assignments[i]
            current_streak_len = 1
        end
        
        # 2. Track how long we've been in the formally committed state
        if assignments[i] == committed_state
            time_in_committed += 1
        end
        
        # 3. Check for a valid transition
        # If we've held a NEW state for at least `min_stable` frames (ignores 1-2-1 flickers)
        if current_streak_state != committed_state && current_streak_len >= min_stable
            
            # Did the PREVIOUS state last long enough to count as a true fixation?
            if time_in_committed >= min_dwell
                # Record transition: it actually started when the current streak began
                transition_index = i - current_streak_len + 1
                push!(transitions, transition_index)
            end
            
            # Commit to the new state and reset the timer
            committed_state = current_streak_state
            time_in_committed = current_streak_len 
        end
    end
    
    return transitions
end
isi_vec = Float64[]
FigNumber = 1
output_dir = "output"

println("Phase 2: Processing individual files...")
for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    
    # 1. Get assignments based on the GLOBAL centers
    assignments = assign_to_centers(x, y, global_centers)
    
    # 2. Extract saccade starts based on our fixed hysteresis rules
    sac_starts = detect_transitions(assignments; min_dwell=1000, min_stable=30)

    # Calculate ISIs (Inter-Saccadic Intervals)
    isi = diff(sac_starts)
    append!(isi_vec, isi)

    # --- Plotting Block (Unchanged) ---
    if length(isi) > 5
        p4 = plot(x, label="X Position", color=:blue, alpha=0.6)
        plot!(y, label="Y Position", color=:green, alpha=0.6)
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
