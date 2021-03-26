using CovidSurvey
using Documenter

makedocs(;
    modules=[CovidSurvey],
    authors="Andreas Koher",
    repo="https://github.com/andreaskoher/CovidSurvey.jl/blob/{commit}{path}#L{line}",
    sitename="CovidSurvey.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andreaskoher.github.io/CovidSurvey.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/andreaskoher/CovidSurvey.jl",
)
