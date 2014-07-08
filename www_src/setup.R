
library("pkg2html")
library("markdown")

pkg <- "mvtnorm"
download.file("http://user.math.uzh.ch/hothorn/TH.bib", dest = "TH.bib")
dest <- "html"
publish <- "../www"

if (!file.exists(dest))
    dir.create(dest)

stopifnot(file.exists(dest)) 

template <- system.file("template", package = "pkg2html")

system(paste("cp -ra", file.path(template, "*"), dest, sep = " "))

wd <- setwd(file.path(dest, "_data"))
R2yaml(pkg)
writeLines(bib2yaml(file.path(wd, "mvtnorm.bib"), 
           c("Genz_Bretz_2009", "Miwa_Hayter_Kuriki_2003")),
            con = "cites.yml")

setwd(wd)
setwd(file.path(dest, "_posts"))
NEWS2md(pkg)

setwd(wd)

Rmd <- list.files(pattern = "Rmd$")

for (f in Rmd)
    writeLines(Rmd2html(f), con = file.path(dest, gsub("Rmd$", "html", f)))

file.remove("TH.bib")

x <- readLines(file.path(dest, "_data", "pkg.yml"))
x <- c(x, "headpic: /img/mvtnorm.png")
writeLines(x, con = file.path(dest, "_data", "pkg.yml"))

file.copy("_config.yml", dest, overwrite = TRUE)

yml <- list.files(pattern = "yml$")
yml <- yml[-grep("^_", yml)]
sapply(yml, function(f) file.copy(f, file.path(dest, "_data"), overwrite = TRUE))

system(paste("cat site.css >> ", file.path(dest, "css", "main.css")))


system(paste("cp ", file.path(publish, "img", "*"), file.path(dest, "img")))

wd <- setwd(dest)


system("jekyll build")

setwd(wd)

system(paste("cp -ra", file.path(dest, "_site/*"), publish))
