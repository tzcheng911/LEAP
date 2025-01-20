library(lme4)
library(lmerTest)
library(tidyverse)
library(dplyr)
library(broom)
library(ggplot2)
library(ggpubr)
library('fastDummies')
library(effsize)
library(lsr)
library(rstatix)
library(car)

################################################################# Load the sensor data
alldata = read.csv("SSEP_sensor.csv")

## Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))

alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="_02")

## Descriptive stats
# sensor
summary_alldata = alldata %>%
  group_by(age,condition) %>%
  summarize(mMeterD = mean(X1.67Hz), mMeterT = mean(X1.11Hz), mBeat = mean(X3.3Hz),Nsubs=n_distinct(sub_id), 
            seMeterD = sd(X1.67Hz)/sqrt(Nsubs),seMeterT = sd(X1.11Hz)/sqrt(Nsubs),seBeat = sd(X3.3Hz)/sqrt(Nsubs))

# need to do log transform to achieve normality
alldata = alldata %>% 
  mutate(log_X1.11Hz = log(X1.11Hz),log_X1.67Hz = log(X1.67Hz), log_X3.3Hz = log(X3.3Hz))

# run ANOVA
duple_random = filter(alldata,condition != "_04")
triple_random = filter(alldata,condition != "_03")

summary(lmer(X1.67Hz ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(X3.3Hz ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(X1.11Hz ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(X3.3Hz ~ 1+ condition*age + (1|sub_id),data=triple_random))

## Visualization
ggplot(duple_random, aes(x = age, y = X1.67Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(triple_random, aes(x = age, y = X1.11Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = age, y = X3.3Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()
