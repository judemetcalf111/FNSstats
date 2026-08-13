using Pkg
Pkg.activate(".")

# using DSP
using SavitzkyGolay
using Random
using StatsBase
using CSV
using DataFrames
using GLM
using Statistics
using Plots
using ImageFiltering
using Distributions
using KernelDensity
using LinearAlgebra

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/StepDistributions/"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 10,
    legendfontsize = 18,
    grid = false,                # Removes visual clutter for smaller plots
    framestyle = :box,           # Professional enclosed bounding box
    dpi = 300,                   # High resolution for PDF export
    margin = 8Plots.mm           # Fixing the label cropping...
)

function process_kde(csv_files)
    num_files = length(csv_files)
    # x_grid represents the log10 of the step sizes
    x_grid = range(-16, 1.6, length=10000)
    total_frequency = zeros(length(x_grid))
    total_points = 0

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        x = df.value1
        y = df.value2
        dt = 0.0001

        dx = diff(x)
        dy = diff(y)

        # 1. Calculate linear speeds first
        speeds = sqrt.(dx.^2 .+ dy.^2) ./ dt

        # 2. Filter out exact zeros to avoid log10(0) = -Inf
        valid_speeds = filter(>(0), speeds)
        n_steps = length(valid_speeds)

        # Skip this file if there are no valid movements
        if n_steps == 0
            continue 
        end

        # 3. Take log10 of the valid speeds
        step_sizes = log10.(valid_speeds)

        # --- Compute KDE and Interpolate ---
        # Note: You can still add `bandwidth=0.01` here if it's too smoothed
        k = kde(step_sizes) 
        ik = InterpKDE(k) 
        
        chunk_density = pdf(ik, x_grid) 
        chunk_frequency = chunk_density .* n_steps

        # 4. Accumulate frequencies directly
        total_frequency .+= chunk_frequency
        total_points += n_steps
        
        # Clean up memory
        x = nothing
        y = nothing
        speeds = nothing
        valid_speeds = nothing
        step_sizes = nothing
        GC.gc() 
        
        println("Processed file $file\n$i / $num_files")
    end

    return x_grid, total_frequency, total_points
end

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

x_grid, total_frequency, total_points = process_kde(csv_files)

# Filter out any 0 frequencies so the log10 y-axis doesn't break
valid_indices = total_frequency .> 0
x_scaled = (4.5/17.6) .* (x_grid .- 1.6) .+ 1.2
final_x = (10 .^ x_scaled)[valid_indices]
final_y = total_frequency[valid_indices]

# Plot the final master curve
final_distribution = plot(final_x, final_y, 
    xscale = :log10, 
    yscale = :log10,
    xlabel = "Step size (rescaled)", 
    ylabel = "Frequency",
    xticks= 10. .^ (-2:1:1),
    xlims = (10^-3,10^1.2),
    linewidth = 4, 
    linecolor = RGB(0.55, 0.71, 0.95),
    grid = true,
    gridalpha = 0.5,
    minorgrid = false,
    legend = false,
    framestyle = :box,
    thickness_scaling = 1.3
)

savefig(final_distribution,joinpath(output_dir, "StepDistribution.pdf"))
