using Pkg
Pkg.activate(".")

using CSV
using DSP
using DataFrames
using GLM
using Statistics
using Plots
using Base.Threads

# Load CSV

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/MAD_Parallel"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 10,
    legendfontsize = 18,
    size = (900, 600),
    grid = false,                # Removes visual clutter for smaller plots
    framestyle = :box,           # Professional enclosed bounding box
    dpi = 300,                   # High resolution for PDF export
    margin = 8Plots.mm           # Fixing the label cropping...
)

# The highly optimized brute-force MAD calculator
function compute_exact_mad(x::Vector{Float64}, max_lag::Int)
    N = length(x)
    mad = zeros(Float64, max_lag + 1)
    
    # We skip lag 0 because the absolute difference of a point with itself is strictly 0.0
    for k in 1:max_lag
        sum_mad = 0.0
        
        # @inbounds turns off safety checks; @simd allows the CPU to calculate multiple iterations at once
        @inbounds @simd for i in 1:(N - k)
            sum_mad += abs(x[i + k] - x[i])
        end
        
        # Average it out for this specific lag
        mad[k + 1] = sum_mad / (N - k)
    end
    
    return mad
end

function calculate_chunked_mad(input_dir, output_dir; K_t::Float64 = 1.5, dt::Float64 = 0.0001)
    isdir(output_dir) || mkdir(output_dir)

    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    
    if tot == 0
        println("No CSV files found in $input_dir")
        return
    end

    max_lag = ceil(Int, K_t / dt)
    
    sum_mad_x = zeros(Float64, max_lag + 1)
    sum_mad_y = zeros(Float64, max_lag + 1)
    valid_files = 0

    println("Calculating exact chunked MAD up to lag $max_lag...")

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        # Force concrete types for maximum speed
        x = Vector{Float64}(df.value1)
        y = Vector{Float64}(df.value2)
        
        if length(x) <= max_lag
            println("Skipping file $i: Not enough data points.")
            continue
        end
        
        # Calculate exactly using our optimized O(N^2) helper function
        mad_x_chunk = compute_exact_mad(x, max_lag)
        mad_y_chunk = compute_exact_mad(y, max_lag)
        
        sum_mad_x .+= mad_x_chunk
        sum_mad_y .+= mad_y_chunk
        valid_files += 1
        
        println("Finished with file $i of $tot")
    end

    if valid_files == 0
        println("Error: No files were long enough to process.")
        return
    end

    # Average the MADs across all valid chunks
    final_mad_x = sum_mad_x ./ valid_files
    final_mad_y = sum_mad_y ./ valid_files
    
    # Total 2D MAD (Adding the 1D components)
    final_mad_total = final_mad_x .+ final_mad_y

    lag_times = range(0, K_t, length=length(final_mad_x))

    # Save to CSV
    out_df = DataFrame(Time = lag_times, MAD_X = final_mad_x, MAD_Y = final_mad_y, MAD_Total = final_mad_total)
    csv_outpath = joinpath(output_dir, "MAD_Exact_Results.csv")
    CSV.write(csv_outpath, out_df)
    
    println("Saved rapid MAD results to $csv_outpath")

    square_final_mad_total = (final_mad_total.^2) .* (10^4)     # Rescale length units to roughly match experimental data
    
    # --- UPDATE PLOT LOGIC ---
    powerlaw_n = 100
    df = DataFrame(X = log.(lag_times[2:powerlaw_n]), Y = log.(square_final_mad_total[2:powerlaw_n]))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    
    plot_sqmad_total = plot(lag_times[2:end] * 1000, square_final_mad_total[2:end],
        linewidth = 12,
        linestyle = :solid,
        color = :steelblue,
        xscale = :log10,
        yscale = :log10,
        grid = true,
        size = (1000, 600),
        dpi = 1200,
        guidefontsize = 30,
        tickfontsize = 30,
        legend = false,
        framestyle = :box,
        # Increase specific margins to accommodate 30pt font
        left_margin = 15Plots.mm, 
        bottom_margin = 15Plots.mm,
        # Keep a smaller margin for top/right if desired
        top_margin = 5Plots.mm,
        right_margin = 5Plots.mm
    )
    
    # Trendline Plot (Start at index 2 to avoid log(0), and scale X by 1000)
    plot!(
        lag_times[2:powerlaw_n] * 1000,
        (lag_times[2:powerlaw_n].^Coeffs[2]) .* (exp(Coeffs[1])), 
        linewidth = 12,
        linestyle = :dash,
        legend = false,
        color = :black
    )
    
    plot!(xlabel = "Time (ms)", ylabel = "Squared MAD", legend = false)
    
    endpoint = 0.02
    tpos = endpoint * 1000 # Matched to the * 1000 scale
    ypos = (endpoint.^Coeffs[2]) .* (exp(Coeffs[1])) .* (10^(-1))
    eqn_string = "t^$(round(Coeffs[2], digits=4))"
    
    annotate!((tpos, ypos, text(eqn_string, :black, 30, :left)))

    savefig(plot_sqmad_total, joinpath(output_dir, "SMAD_Final.pdf"))
    println("Saved rendered plot to SMAD_Final.pdf")
end

calculate_chunked_mad(input_dir, output_dir; K_t=3., dt=0.0001*0.023)
