using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Base.Threads

# Load CSV

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/MSD"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

# --- 2. Define the MSD Function ---
function calculate_msd(x, y,n)
    N = length(x)
    msd_result = Float64[]
    
    # Loop through time lags (tau) from 1 up to half the trajectory length
    # (Going beyond N/2 usually makes the statistics too noisy)
    for tau in 1:n
        # Create shifted arrays to calculate displacement
        # x(t + tau) - x(t)
        dx = x[1+tau:end] .- x[1:end-tau]
        dy = y[1+tau:end] .- y[1:end-tau]
        
        # Squared displacement: dx^2 + dy^2
        sq_displacement = dx.^2 .+ dy.^2
        
        # Mean of the squared displacements for this lag
        push!(msd_result, mean(sq_displacement))
    end
    
    return msd_result
end

for file in csv_files
    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    # Extract the columns (x = value1, y = value2)
    x = df.value1
    y = df.value2

    n = 15000
    t = 0.001 * collect(1:n)
    xtest = x[1:end-n]
    m = length(xtest)

    # --- 3. Run and Print ---
    msd_values = calculate_msd(x, y, n)

    println("Calculation Complete.")
    println("First 5 MSD values:")
    for i in 1:min(5, length(msd_values))
        println("Lag $i: $(msd_values[i])")
    end

    # Update DataFrame Logic
    df = DataFrame(X = log.(t), Y = log.(msd_values))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    
    plot(
        t,
        msd_values,
        label = "MSD",             # Renamed from Means
        linewidth = 5,
        linestyle = :solid,
        color = :Orange,
        xscale = :log10,
        yscale = :log10,
        grid = true,
        size = (800, 600),           # Increases the base resolution canvas
        dpi = 1200,                   # High resolution for crisp lines and text
        guidefontsize = 20,          # Increases axis label font size (xlabel, ylabel)
        tickfontsize = 12,           # Increases axis number/tick font size
        legendfontsize = 20,         # Increases legend text size
        framestyle = :box,           # Adds a clean border around the entire plot
        margin = 5Plots.mm           # Prevents larger labels from being cut off
    )

    plot!(
        t,
        (t.^Coeffs[2]) .* (exp(Coeffs[1])), # slightly cleaner exp() syntax
        label = "Trendline",
        linewidth = 3,
        linestyle = :dash,
        color = :Black
    )

    # Update axis labels for MSD
    plot!(xlabel = "Time", ylabel = "MSD ⟨r²⟩", legend = :outerright)

    endpoint = 0.005

    tpos = endpoint                 # choose a spot
    ypos = (endpoint.^Coeffs[2]) .* (ℯ.^Coeffs[1]) .* (10^2)   # y-value to match scale
    eqn_string = "⟨r²⟩ ~ t^$(round(Coeffs[2], digits=3))"


    annotate!((tpos, ypos, text(eqn_string, :Black, 16, :left)))

    outname = joinpath(output_dir, splitext(basename(file))[1] * "-MDS" * ".pdf")

    # Save figure

    savefig(outname)

    println("Fig number: $FigNumber" )
    global FigNumber = FigNumber + 1
end
