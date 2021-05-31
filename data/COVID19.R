library("COVID19")
gmr <- "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
x <- covid19("DK", level=1, gmr = gmr, start="2020-02-01")
saveRDS(x, file="~/dev/CovidSurvey/data/COVID19/DK.rds")
