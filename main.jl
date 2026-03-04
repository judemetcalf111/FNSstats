using Pkg
Pkg.activate(".")

using CSV, DataFrames, Plots, Measures, LaTeXStrings

if length(ARGS) != 1
    println("Usage: julia main.jl <filename.csv>")
    exit(1)
end

filename = ARGS[1]
output_name = splitext(basename(filename))[1]

# Read DataFrame
df = CSV.read(filename, DataFrame)

# Extract Column Names and Data
col_names = names(df)
x_col = col_names[1]
y_col = col_names[2]

# Filter out zero or negative values for the log plot to avoid errors
# (Assuming Frequency/Magnitude data where 0Hz exists)
xs = df[:, 1]
ys = df[:, 2]

# --- Linear Regression ---
superdiff_idx = findfirst(ys .> 5)

logxs = log10.(df[2:superdiff_idx, 1])
logys = log10.(df[2:superdiff_idx, 2])


X = [logxs ones(length(logxs))]  # Design matrix with intercept
β = X \ logys                    # Solve for coefficients (slope and intercept)
slope, intercept = β
println("Linear Fit: y = $(round(slope, sigdigits=3)) * x + $(round(intercept, sigdigits=3))")

# --- PUBLICATION STYLING ---
default(
    fontfamily = "Computer Modern", # or "Helvetica" / "Arial"
    linewidth = 6,                  # Thick lines
    framestyle = :box,              # Closed box around plot
    label = nothing,                # Hide legend unless specified
    grid = :true,
    gridstyle = :dash,
    gridalpha = 0.3,
    tickfontsize = 18,
    guidefontsize = 18,
    titlefontsize = 16,
    margin = 5mm                    # Requires 'using Measures'
)

# # --- PLOT 1: LINEAR SCALE ---
# p1 = plot(xs, ys,
#     xlabel = x_col,
#     ylabel = y_col,
#     title = "Linear Scale",
#     linecolor = :navy
# )

# --- PLOT 2: LOG-LOG SCALE ---
# We filter x > 0 for the log plot to avoid -Inf errors
mask = (xs .> 0) .& (ys .> 0)

p = plot(xs[mask], ys[mask],
    xlabel = x_col,
    ylabel = y_col,
    # title = "Log-Log Scale",
    xscale = :log10,              # Let Plots handle the log scale for better ticks
    yscale = :log10,
    linecolor = :darkred,
    # minorgrid = true
    ylims = (0.001, 1000)
)

p = plot!(p, 10 .^ logxs, 10 .^ (slope .* logxs .+ intercept),
    label = "Fit: y = $(round(slope, sigdigits=3))x + $(round(intercept, sigdigits=3))",
    xscale = :log10,              # Let Plots handle the log scale for better ticks
    yscale = :log10,
    linecolor = :black,
    linestyle = :dash
)

# --- COMBINE AND SAVE ---
# layout=(1,2) creates 1 row, 2 columns
final_plot = plot(p, size = (600, 500))

savefig(final_plot, output_name * ".pdf")
println("High-quality plot saved as $output_name.pdf")
