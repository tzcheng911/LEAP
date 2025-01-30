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
alldataL = read.csv("lAM_roi_redo10_conn_plv.csv")
alldataR = read.csv("rAM_roi_redo10_conn_plv.csv")
alldata = read.csv("roi_redo5_conn_GC_MA.csv")
alldata = read.csv("roi_redo5_conn_GC_M_seed_MI.csv")
alldata = read.csv("conn_plv_roi_redo4_SA.csv")

# log transform to achieve normality 
alldata = alldata %>% 
  mutate(log_Delta.conn = log(Delta.conn),log_Theta.conn = log(Theta.conn), log_Alpha.conn = log(Alpha.conn), log_Beta.conn = log(Beta.conn))

# Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))

alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="_02")


## Descriptive stats
# Connectivity
summary_alldata = alldata %>%
  group_by(age,condition) %>%
  summarize(mDeltaConn = mean(Delta.conn), mThetaConn = mean(Theta.conn), mAlphaConn = mean(Alpha.conn), mBetaConn = mean(Beta.conn),Nsubs=n_distinct(sub_id), 
            seDeltaConn = sd(Delta.conn)/sqrt(Nsubs),seThetaConn = sd(Theta.conn)/sqrt(Nsubs),seAlphaConn = sd(Alpha.conn)/sqrt(Nsubs),seBetaConn = sd(Beta.conn)/sqrt(Nsubs))

################################################################# run ANOVA
duple_random = filter(alldata,condition != "_04")
triple_random = filter(alldata,condition != "_03")

summary(lmer(Delta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Theta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Alpha.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Beta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))

# notes for duple: even before log transform residual is pretty normal for theta, alpha and beta
lm = lmer(Theta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random)
hist(residuals(lm))

summary(lmer(Delta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Theta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Alpha.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Beta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))

# notes for triple: even before log transform residual is pretty normal for theta, alpha and beta
lm = lmer(log_Alpha.conn ~ 1+ condition*age + (1|sub_id),data=triple_random)
hist(residuals(lm))

summary(lmer(Delta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Theta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Alpha.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))
summary(lmer(Beta.conn ~ 1+ condition*age + (1|sub_id),data=duple_random))

summary(lmer(Delta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Theta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Alpha.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))
summary(lmer(Beta.conn ~ 1+ condition*age + (1|sub_id),data=triple_random))

## Visualization
ggplot(alldata, aes(x = age, y = Theta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.1,dodge.width = 0.9), color="black")+
  ylim(0,1)+
  theme_bw()

ggplot(alldata, aes(x = age, y = Alpha.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.1,dodge.width = 0.9), color="black")+
  ylim(0,1)+
  theme_bw()

# ggplot(duple_random, aes(x = age, y = Delta.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()

ggplot(duple_random, aes(x = age, y = Theta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(duple_random, aes(x = age, y = Alpha.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

# ggplot(duple_random, aes(x = age, y = Beta.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()
# 
# ggplot(duple_random, aes(x = age, y = Broadband.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()
# 
# ggplot(triple_random, aes(x = age, y = Delta.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()

ggplot(triple_random, aes(x = age, y = Theta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(triple_random, aes(x = age, y = Alpha.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

# ggplot(triple_random, aes(x = age, y = Beta.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()
# 
# ggplot(triple_random, aes(x = age, y = Broadband.conn, fill = condition)) +
#   geom_bar(stat="summary", position='dodge') +
#   stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
#   geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
#   theme_bw()

