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

################################################################# Load the ROI data
alldata = read.csv("SSEP_roi5.csv")

# recode to collapse left and right ROIs if needed ******be careful to overwrite******
# pool the 5 ROIs from 10 ROIs
alldata = alldata %>%
  mutate(ROI =fct_collapse(ROI,Auditory = c("AuditoryL", "AuditoryR"),Motor = c("MotorL", "MotorR"),
                                  Sensory = c("SensoryL","SensoryR"),BG = c("BGL","BGR"), IFG = c("IFGL","IFGR")))
alldata = alldata %>%
  group_by(sub_id,age,condition,ROI) %>%
  summarize(X1.67Hz = mean(X1.67Hz), X1.11Hz = mean(X1.11Hz), X3.3Hz = mean(X3.3Hz),Nsubs=n_distinct(sub_id), .groups = "drop") 

# pool the 5 ROIs from 114 ROIs
alldata = alldata %>%
  mutate(ROI =fct_collapse(ROI,Auditory = c("AuditoryL", "AuditoryR"),Motor = c("MotorL", "MotorR"),
                           Sensory = c("SensoryL","SensoryR"),BG = c("BGL","BGR"), IFG = c("IFGL","IFGR")))
alldata %>%
  group_by(sub_id,age,condition,ROI) %>%
  summarize(X1.67Hz = mean(X1.67Hz), X1.11Hz = mean(X1.11Hz), X3.3Hz = mean(X3.3Hz),Nsubs=n_distinct(sub_id), .groups = "drop") 

# log transform to achieve normality
alldata = alldata %>% 
  mutate(log_X1.11Hz = log(X1.11Hz),log_X1.67Hz = log(X1.67Hz), log_X3.3Hz = log(X3.3Hz))

# Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(ROI = as.factor(ROI))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))

alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="_02")

alldata = alldata %>%
  mutate(nlog_X1.11Hz = scale(log_X1.11Hz, center=TRUE, scale=FALSE))
alldata$X1.67Hz = scale(log_X1.67Hz, center=TRUE, scale=FALSE)
alldata$X3.3Hz = scale(log_X3.33Hz, center=TRUE, scale=FALSE)

## Descriptive stats
# SSEP
summary_alldata = alldata %>%
  group_by(age,condition,ROI) %>%
  summarize(mMeterD = mean(X1.67Hz), mMeterT = mean(X1.11Hz), mBeat = mean(X3.3Hz),Nsubs=n_distinct(sub_id), 
            seMeterD = sd(X1.67Hz)/sqrt(Nsubs),seMeterT = sd(X1.11Hz)/sqrt(Nsubs),seBeat = sd(X3.3Hz)/sqrt(Nsubs))

################################################################# run ANOVA
which_ROI = "Motor" # AuditoryL,AuditoryR,MotorL,MotorR,SensoryL,SensoryR,BGL,BGR,IFGL,IFGR, or the recode ROI without L/R

duple_random = filter(alldata,condition != "_04", ROI == which_ROI)
triple_random = filter(alldata,condition != "_03", ROI == which_ROI)
summary(lmer(X1.67Hz ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(X3.3Hz ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(X1.11Hz ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(X3.3Hz ~ 1+ condition*age + (1|sub_id),data=triple_random))

## Visualization
ggplot(duple_random, aes(x = age, y = X1.67Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#  scale_y_continuous(trans = "log", breaks=c(0,0.25,0.35,0.5,1,5,8)) +
#  ylim(0,2.5)+
  theme_bw()

ggplot(duple_random, aes(x = age, y = X3.3Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  #  ylim(0,2)+
  theme_bw()

ggplot(triple_random, aes(x = age, y = X1.11Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#  ylim(0,2.5)+
  theme_bw()

ggplot(triple_random, aes(x = age, y = X3.3Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#  ylim(0,2)+
  theme_bw()

## experiment LMM
summary_duple_random = duple_random %>%
  group_by(condition,age) %>%
  summarize(mMeter = mean(X1.67Hz))

summary(lmer(X1.67Hz ~ 1+ condition + (1|sub_id),data=duple_random))
summary(lmer(X1.67Hz ~ 1+ condition + age + (1|sub_id),data=duple_random))
summary(lmer(X1.67Hz ~ 1+ condition*age + (1|sub_id),data=duple_random))

