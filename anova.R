library(dplyr)

fname <- 'Surprise averaged responses.csv'

data <- read.csv(fname)
data$cluster <- ordered(data$cluster, levels = c(0, 1, 2, 3, 4))
data$student_id <- NULL
res.aov <- aov(response ~ cluster, data = data)
summary(res.aov)