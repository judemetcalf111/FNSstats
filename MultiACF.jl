using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Plots

function calculate_walker_acf(input_dir, output_dir; K_t::Float64 = 1.5, scaling::Float64=1.0)
    # Make sure output folder exists
    isdir(output_dir) || mkdir(output_dir)

    # Get all .csv files in the input directory
    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    
    if tot == 0
        println("No CSV files found in $input_dir")
        return
    end

    dt = 0.001
    K = ceil(Int, K_t/(dt*scaling))
    
    println("Pass 1: Calculating global means...")
    sum_x = 0.0
    sum_v = 0.0
    N_x = 0
    N_v = 0

    for file in csv_files
        df = CSV.read(file, DataFrame)
        
        x = df.value1
        y = df.value2
        
        # Calculate velocity
        vx = diff(x) ./ dt
        vy = diff(y) ./ dt
        v = sqrt.(vx.^2 .+ vy.^2)
        
        sum_x += sum(x)
        N_x += length(x)
        
        sum_v += sum(v)
        N_v += length(v)
    end

    mu_x = sum_x / N_x
    mu_v = sum_v / N_v

    println("Pass 2: Calculating cross-file ACF up to lag $K...")
    
    # Arrays to hold the sum of products for lag 0 to K
    # Index 1 = Lag 0 (Variance), Index 2 = Lag 1, etc.
    lag_sums_x = zeros(Float64, K + 1)
    lag_sums_v = zeros(Float64, K + 1)
    
    # Buffers to carry over the last K points from the previous file
    buffer_x = Float64[]
    buffer_v = Float64[]

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        x_raw = df.value1
        y_raw = df.value2
        
        vx = diff(x_raw) ./ dt
        vy = diff(y_raw) ./ dt
        v_raw = sqrt.(vx.^2 .+ vy.^2)
        
        # Mean-center the data using the GLOBAL mean
        x = x_raw .- mu_x
        v = v_raw .- mu_v
        
        # Prepend the buffer from the previous file (empty on the very first file)
        x_padded = vcat(buffer_x, x)
        v_padded = vcat(buffer_v, v)
        
        # Calculate lagged products
        for k in 0:K
            # We only iterate over the *new* data points from the current file, 
            # but we look backwards into the buffer for the lagged values.
            start_idx = length(buffer_x) + 1
            
            for j in start_idx:length(x_padded)
                if j - k > 0 # Prevents out-of-bounds on the very first file
                    lag_sums_x[k+1] += x_padded[j] * x_padded[j - k]
                end
            end
            
            for j in (length(buffer_v) + 1):length(v_padded)
                if j - k > 0
                    lag_sums_v[k+1] += v_padded[j] * v_padded[j - k]
                end
            end
        end
        
        # Update buffers with the last K points of the current file
        # The max() ensures it won't crash if a file happens to be shorter than K
        buffer_x = x[max(1, end-K+1):end]
        buffer_v = v[max(1, end-K+1):end]
    end

    # Calculate final ACF (divide by lag 0 sum, which is the total variance)
    acf_x = lag_sums_x ./ lag_sums_x[1]
    acf_v = lag_sums_v ./ lag_sums_v[1]

    lag = range(0,K_t,length(acf_x))

    # Save to CSV
    out_df = DataFrame(Lag = lag, ACF_X = acf_x, ACF_V = acf_v)
    csv_outpath = joinpath(output_dir, "ACF_Results.csv")
    CSV.write(csv_outpath, out_df)
    
    println("Saved ACF results to $csv_outpath")
    
    # Optional: Quick terminal output for K=1
    println("---")
    println("ACF(X) at lag 1: ", round(acf_x[2], digits=5))
    println("ACF(V) at lag 1: ", round(acf_v[2], digits=5))

    default(
        fontfamily = "Helvetica",    # Clean, publication-ready font
        titlefontsize = 12,
        guidefontsize = 11,          
        tickfontsize = 9,
        legendfontsize = 9,
        grid = false,                # Removes visual clutter
        framestyle = :box,           # Professional enclosed bounding box
        dpi = 300                    # High resolution for PDF export
    )
    
    plot_acf_x = plot(
        lag .* 1000, acf_x,
        label = "Position ACF", 
        linewidth = 2, 
        color = :steelblue,
        xlabel = "Lag (ms)",
        ylabel = "Autocorrelation"
    )
    savefig(plot_acf_x, joinpath(output_dir, "ACF_x.pdf"))

    plot_acf_v = plot(
        lag .* 1000, acf_v,
        label = "Velocity ACF", 
        linewidth = 2, 
        color = :steelblue,
        xlabel = "Lag (ms)",
        ylabel = "Autocorrelation"
    )
    savefig(plot_acf_v, joinpath(output_dir, "ACF_v.pdf"))

    return out_df
end

input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ACFs"
K_t = 1.5
calculate_walker_acf(input_dir, output_dir, K_t=K_t, scaling=0.3)
