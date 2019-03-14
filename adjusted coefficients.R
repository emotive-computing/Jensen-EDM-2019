library(dplyr)
library(broom)
library(QuantPsyc)

df = read.csv("averaged valence all.csv")
dfHour = df %>% group_by(survey_question) %>% 
  do(fitHour = lm(survey_answer ~ bio_video_watch + video_play + karma_awarded + leaderboard_load + personal_profile_picture + tys_answer + tys_finish + tys_load + tys_previous + tys_review_correct_question + tys_review_incorrect_question + tys_review_solution_video + tys_review_topic_video + tys_unload + video_caption + video_completed + video_play + video_pause + video_seek + video_watch + wall_load_more + wall_make_post + wall_page_load, data=.))
dfHourCoef = tidy(dfHour, fitHour)
p= dfHourCoef$p.value
adjusted = p.adjust(p, method = "fdr", n = length(p))
dfHourCoef['adjusted'] <- adjusted
write.csv( dfHourCoef, "betaAll.csv")
