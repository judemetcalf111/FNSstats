using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using StatsBase
using Plots

using CSV, DataFrames, StatsBase

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 24,          
    tickfontsize = 18,
    legendfontsize = 20,
    grid = false,                # Removes visual clutter for smaller plots
    framestyle = :box,           # Professional enclosed bounding box
    dpi = 300,                   # High resolution for PDF export
    margin = 8Plots.mm           # Fixing the label cropping...
)

function calculate_fast_chunked_acf(input_dir, output_dir; K_t::Float64 = 1.5, scaling::Float64=1.0)
    isdir(output_dir) || mkdir(output_dir)

    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    
    if tot == 0
        println("No CSV files found in $input_dir")
        return
    end

    dt = 0.001
    K = ceil(Int, K_t / (dt * scaling))
    
    println("Calculating chunked FFT ACF up to lag $K...")
    
    # Preallocate the matrix for maximum possible files
    acf_x_matrix = zeros(Float64, K + 1, tot)
    # sum_acf_v = zeros(Float64, K + 1)
    valid_files = 0

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        x = Vector{Float64}(df.value1)
        # y = Vector{Float64}(df.value2)
        
        # Edge case check: skip files shorter than our max lag
        if length(x) <= K
            println("Skipping file $i: Not enough data points for lag $K")
            continue
        end
        
        acf_x_chunk = autocor(x, 0:K)
        
        # Increment valid files counter BEFORE assignment
        valid_files += 1
        
        # Assign the chunk to the next available valid column
        acf_x_matrix[:, valid_files] = acf_x_chunk
        
        println("Finished with file $i of $tot")
    end

    if valid_files == 0
        println("Error: No files were long enough to process.")
        return
    end

    # CRITICAL: Trim the matrix to drop the empty zero columns from skipped files
    acf_x_matrix = acf_x_matrix[:, 1:valid_files]

    # Average the ACFs across all valid chunks
    final_acf_x = vec(mean(acf_x_matrix, dims=2))
    std_acf_x = vec(std(acf_x_matrix,dims=2))
    sem_acf_x = std_acf_x ./ sqrt(valid_files)
    # final_acf_v = sum_acf_v ./ valid_files

    lag = range(0, K_t, length=length(final_acf_x))

    # Save to CSV
    out_df = DataFrame(Lag = lag, ACF_X = final_acf_x, SEM_ACF_X = sem_acf_x)#, ACF_V = final_acf_v)
    csv_outpath = joinpath(output_dir, "ACF_Results_Fast.csv")
    CSV.write(csv_outpath, out_df)
    
    println("Saved rapid ACF results to $csv_outpath")
    
    println("---")
    println("ACF(X) at lag 1: ", round(final_acf_x[2], digits=5))
    # println("ACF(V) at lag 1: ", round(final_acf_v[2], digits=5))

    plot_acf_x = plot(
        lag .* 1000, final_acf_x,
        ribbon = 5 .* sem_acf_x,     # Automatically plots shading above and below the line
        fillalpha = 0.3,        # Controls the transparency of the shaded SEM band
        label = "X-Position ACF", 
        linewidth = 4, 
        color = :steelblue,
        xlabel = "Lag (ms)",
        ylabel = "Autocorrelation"
    )
    savefig(plot_acf_x, joinpath(output_dir, "ACF_x.pdf"))

    # plot_acf_v = plot(
    #     lag .* 1000, final_acf_v,
    #     label = "Velocity ACF", 
    #     linewidth = 2, 
    #     color = :steelblue,
    #     xlabel = "Lag (ms)",
    #     ylabel = "Autocorrelation"
    # )
    # savefig(plot_acf_v, joinpath(output_dir, "ACF_v.pdf"))

    return out_df
end

input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ACFs"
K_t = 1.5
calculate_fast_chunked_acf(input_dir, output_dir, K_t=K_t, scaling=0.023)
