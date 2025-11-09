using Documenter
using GAM

makedocs(
    sitename = "GAM.jl Documentation",
    modules = [GAM],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api_reference.md",
    ],
    format = Documenter.HTML(),
    authors = "Your Name",
    remotes = nothing
)