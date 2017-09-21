setwd('../Desktop/Project/Ride Time Prediction/')

library(data.table)
library(caret)
library(lubridate)
library(ggplot2)
library(ggmap)
library(geosphere)
library(rvest)
library(dplyr)
library(tibble)
library(forcats)
library(stringr)
library(corrplot)
library(xgboost)
library(readr)

rm(list=ls())

train <- as.tibble(fread('train.csv'))
test <- as.tibble(fread('test.csv'))

weather <- as.tibble(fread('weather_data_nyc_centralpark_2016.csv'))

foo <- as.tibble(fread('fastest_routes_train_part_1.csv'))
bar <- as.tibble(fread('fastest_routes_train_part_2.csv'))
fastest_route <- bind_rows(foo, bar)
rm(foo,bar)

combine <- bind_rows(train %>% mutate(dset = 'train'),
                     test %>% mutate(dset = 'test', 
                                     dropoff_datetime = NA,
                                     trip_duration = NA))
combine <- combine %>% mutate(dset = factor(dset))

#################################Data Cleaning#######################################

#remove records with taxis of same pickup/dropoff
train <- train %>% mutate(dif_lat = pickup_latitude - dropoff_longitude,
                          dif_lon = pickup_longitude - dropoff_longitude)
train <- subset(train, abs(dif_lat) > 0 | abs(dif_lon) > 0)

#remove records with coordinates outside NY city
long_limit <- c(-74.03, -73.77)
lat_limit <- c(40.63,40.85)

train <- subset(train, train$pickup_longitude > long_limit[1] & train$pickup_longitude < long_limit[2])
train <- subset(train, train$dropoff_longitude > long_limit[1] & train$dropoff_longitude < long_limit[2])
train <- subset(train, train$pickup_latitude > lat_limit[1] & train$pickup_latitude < lat_limit[2])
train <- subset(train, train$dropoff_latitude > lat_limit[1] & train$dropoff_latitude < lat_limit[2])

#Changing time variables to standard format
train <- train %>%
  mutate(pickup_datetime = ymd_hms(pickup_datetime),
         dropoff_datetime = ymd_hms(dropoff_datetime),
         vendor_id = factor(vendor_id),
         passenger_count = factor(passenger_count))

#################################Feature Engineering###################################

#Creating new features using spatial clusters based on dropoff/pickup
jfk_coord <- tibble(lon = -73.778889, lat = 40.639722)
la_guardia_coord <- tibble(lon = -73.872611, lat = 40.77725)

pick_coord <- train %>% select(pickup_longitude, pickup_latitude)
drop_coord <- train %>% select(dropoff_longitude, dropoff_latitude)

train$Cdist <- distCosine(pick_coord, drop_coord)
train$bearing <- bearing(pick_coord, drop_coord)

train$jfk_dist_pick <- distCosine(pick_coord, jfk_coord)
train$jfk_dist_drop <- distCosine(drop_coord, jfk_coord)
train$lg_dist_pick <- distCosine(pick_coord, la_guardia_coord) 
train$lg_dist_drop <- distCosine(drop_coord, la_guardia_coord)

pick_clus <- kmeans(pick_coord, 15)
drop_clus <- kmeans(drop_coord, 15)

train$pick_clus <- pick_clus$cluster
train$drop_clus <- drop_clus$cluster

#Extract time and day features#
train <- train %>% 
  mutate(speed = Cdist/trip_duration*3.6,
         date = date(pickup_datetime),
         month = month(pickup_datetime, label = TRUE),
         wday = wday(pickup_datetime, label = TRUE),
         wday = fct_relevel(wday, c('Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun')),
         hour = hour(pickup_datetime),
         work = (hour %in% seq(8,18)) & (wday %in% c('Mon', 'Tues', 'Wed', 'Thurs', 'Fri')),
         jfk_trip = (jfk_dist_pick < 2e3) | (jfk_dist_drop < 2e3),
         lg_trip = (lg_dist_pick < 2e3) | (lg_dist_drop < 2e3),
         blizzard = !((date < ymd('2016-01-22') | (date > ymd('2016-01-29'))) )
         )

#adding weather data
weather <- weather %>%
  mutate(date = dmy(date),
         rain = as.numeric(ifelse(precip == "T", "0.01", precip)),
         s_fall = as.numeric(ifelse(snowfall == "T", "0.01", snowfall)),
         s_depth = as.numeric(ifelse(snowdepth == "T", "0.01", snowdepth)),
         all_precip = s_fall + rain,
         has_snow = (s_fall > 0) | (s_depth > 0),
         has_rain = rain > 0)

train <- train %>% left_join(weather, by = 'date')

train <- train %>%
  filter(trip_duration < 22*3600,
         Cdist > 0 | (near(Cdist, 0) & trip_duration < 60),
         jfk_dist_pick < 3e5 & jfk_dist_drop < 3e5,
         trip_duration > 10,
         speed < 100)

#Fastest Routes
new_foo <- fastest_route %>%
  select(id, total_distance, total_travel_time, number_of_steps,
         step_direction, step_maneuvers) %>%
  mutate(fastest_speed = total_distance/total_travel_time*3.6,
         left_turns = str_count(step_direction, 'left'),
         right_turns = str_count(step_direction, 'right'),
         turns = str_count(step_maneuvers, 'turn')
         ) %>%
  select(-step_direction, -step_maneuvers)

train <- left_join(train, new_foo, by ='id') %>%
  mutate(fast_speed_trip = total_distance/trip_duration*3.6)


#CorrelationPlot
train %>%
  select(-id, -pickup_datetime, -dropoff_datetime, -jfk_dist_pick, -jfk_dist_drop, 
         -lg_dist_pick, -lg_dist_drop, -date, -pickup_longitude, -pickup_latitude,
         -dropoff_longitude,-dropoff_latitude, -dif_lat, -dif_lon) %>%
  mutate(passenger_count = as.integer(passenger_count),
         vendor_id = as.integer(vendor_id),
         store_and_fwd_flag = as.integer(as.factor(store_and_fwd_flag)),
         month = as.integer(as.factor(month)),
         wday = as.integer(as.factor(wday)),
         work = as.integer(as.factor(work)),
         Cdist = as.integer(Cdist),
         bearing = as.integer(bearing),
         Holiday = as.integer(Holiday),
         jfk_trip = as.integer(jfk_trip),
         lg_trip = as.integer(lg_trip),
         blizzard = as.integer(blizzard),
         max_temp = as.integer(max_temp),
         min_temp = as.integer(min_temp),
         avg_temp = as.integer(avg_temp),
         event1 = as.integer(event1),
         event2 = as.integer(event2),
         event3 = as.integer(event3),
         speed = as.integer(speed),
         total_distance = as.integer(total_distance),
         total_travel_time = as.integer(total_travel_time)) %>%
  select(trip_duration, speed, everything()) %>% 
  cor(use='complete.obs', method = 'spearman') %>%
  corrplot(type='lower', method='circle', diag=F)





#####ModelingStep#####
#Combining Data
jfk_coord <- tibble(lon = -73.778889, lat = 40.639722)
la_guardia_coord <- tibble(lon = -73.872611, lat = 40.77725)

#adding distance variables
pick_coord <- combine %>%
  select(pickup_longitude, pickup_latitude)
drop_coord <- combine %>%
  select(dropoff_longitude, dropoff_latitude)
combine$dist <- distCosine(pick_coord, drop_coord)
combine$bearing <- bearing(pick_coord, drop_coord)

combine$jfk_dist_pick <- distCosine(pick_coord, jfk_coord)
combine$jfk_dist_drop <- distCosine(drop_coord, jfk_coord)
combine$lg_dist_pick <- distCosine(pick_coord, la_guardia_coord)
combine$lg_dist_drop <- distCosine(drop_coord, la_guardia_coord)

#cleaning datetime data format
combine <- combine %>%
  mutate(pickup_datetime = ymd_hms(pickup_datetime),
         dropoff_datetime = ymd_hms(dropoff_datetime),
         date = date(pickup_datetime))

#adding weather data
#run weather preprocessing before this step
foo <- weather %>% select(date, rain, s_fall, s_depth, all_precip, has_snow, has_rain, max_temp, min_temp)
combine <- left_join(combine, foo, by = "date")

#adding fast routes data
fast <- fastest_route %>%
  select(id, total_distance, total_travel_time, number_of_steps,
         step_direction, step_maneuvers) %>%
  mutate(fastest_speed = total_distance/total_travel_time*3.6,
         left_turns = str_count(step_direction, "left"),
         right_turns = str_count(step_direction, "right"),
         turns = str_count(step_maneuvers, "turn")
  ) %>%
  select(-step_direction, -step_maneuvers)
combine <- left_join(combine, fast, by = "id")

#Final check for data format
combine <- combine %>%
  mutate(store_and_fwd_flag = as.integer(as.factor(store_and_fwd_flag)),
         vendor_id = as.integer(vendor_id),
         month = as.integer(month(pickup_datetime)),
         hour = as.integer(hour(pickup_datetime)),
         wday = wday(pickup_datetime, label = TRUE),
         wday = as.integer(fct_relevel(wday, c("Sun", "Sat", "Mon", "Tues", "Wed", "Thurs", "Fri"))),
         work = as.integer((hour %in% seq(8,18)) & (wday %in% c("Mon", "Tues", "Wed", "Thurs", "Fri"))),
         jfk_trip = as.integer((jfk_dist_pick < 2e3) | (jfk_dist_drop < 2e3)),
         lg_trip = as.integer((lg_dist_pick < 2e3) | (lg_dist_drop < 2e3)),
         has_rain = as.integer(has_rain),
         has_snow = as.integer(has_snow),
         blizzard = as.integer(!( (date < ymd("2016-01-22") | (date > ymd("2016-01-29")))))
  )

glimpse(combine)

# predictor features
train_cols <- c("total_travel_time", "total_distance", "hour", "dist",
                "vendor_id", "jfk_trip", "lg_trip", "wday", "month",
                "pickup_longitude", "pickup_latitude", "bearing", "lg_dist_drop")
# target feature
y_col <- c("trip_duration")
# identification feature
id_col <- c("id") 
# auxilliary features
aux_cols <- c("dset")
# cleaning features
clean_cols <- c("jfk_dist_drop", "jfk_dist_pick")

test_id <- combine %>%
  filter(dset == "test") %>%
  select_(.dots = id_col)
cols <- c(train_cols, y_col, aux_cols, clean_cols)
combine <- combine %>%
  select_(.dots = cols)

# split train/test
train <- combine %>%
  filter(dset == "train") %>%
  select_(.dots = str_c("-",c(aux_cols)))
test <- combine %>%
  filter(dset == "test") %>%
  select_(.dots = str_c("-",c(aux_cols, clean_cols, y_col)))

train <- train %>%
  mutate(trip_duration = log(trip_duration + 1))

set.seed(2017)
trainIndex <- createDataPartition(train$trip_duration, p = 0.8, list = F, times = 1)

train <- train[trainIndex,]
valid <- train[-trainIndex,]

valid <- valid %>%
  select_(.dots = str_c("-", c(clean_cols)))

train <- train %>%
  filter(trip_duration < 24*3600,
         jfk_dist_pick < 3e5, jfk_dist_drop < 3e5) %>%
  select_(.dots = str_c("-", c(clean_cols)))

foo <- train %>% select(-trip_duration)
bar <- valid %>% select(-trip_duration)

dtrain <- xgb.DMatrix(as.matrix(foo), label = train$trip_duration)
dvalid <- xgb.DMatrix(as.matrix(bar), label = valid$trip_duration)
dtest <- xgb.DMatrix(as.matrix(test))

xgb_param <- list(colsample_bytree = 0.7,
                  subsample = 0.7,
                  booster = "gbtree",
                  max_depth = 5,
                  eta = 0.3, #shrinkage
                  eval_metric = "rmse", 
                  objective = "reg:linear",
                  seed = 2017)

watchlist <- list(train = dtrain, valid = dvalid)

gb_fit1 <- xgb.train(params = xgb_param,
                     data = dtrain,
                     watchlist = watchlist,
                     nrounds = 200,
                     print_every_n = 5)

xgb.cv <- xgb.cv(params = xgb_param,
                 data = dtrain,
                 nrounds = 100,
                 nfold = 5,
                 showsd = TRUE,
                 verbose = T,
                 print_every_n =3,
                 early_stopping_rounds = 10)

xgb.cv$

#Variable Importance Plot
imp_matrix <- as.tibble(xgb.importance(feature_names = colnames(train %>% select(-trip_duration)), 
                                       model = gb_fit1))


imp_matrix %>%
  ggplot(aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")

test_pred <- predict(gb_fit1, dtest)
pred <- test_id %>% mutate(trip_duration = exp(test_pred - 1))

pred %>% write_csv('submit.csv')
