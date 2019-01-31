# Conceptual Model

## Requirements

- The user wants to explore the impact of severe weather on the frequencies and severities of accidents.

- In addition, the users are wanted to identify neighborhoods, streets and intersections where accidents often occur

- The users are able to explore the trends in types and frequencies of accidents over  the  four  years,  notably  the  trends  in  numbers of fatal injuries.

## Accident_Type Dimension

| __Accident_Type_key__ | Impact_Type | Traffic_controll | Road_Condition | is_fatal |
| --------------------- | ----------- | ---------------- | -------------- | -------- |
| int32                 | enum[0-99]  | enum[0-10]       | enum[0- 7]     | eum[0-1] |

- **Description:** This Dimension supports the user to explore the types of accident in terms of road condition , its impact and that given road condition. The motivation is support query filtering based on this columns to reduce column sizes in Fact Table.

## Date Dimension

| __Date_key__ | Year   | Month      | Day        | Week_of_Day_Key | Day_of_the_week | Time_of_the_day | Special_day  |
| ------------ | ------ | ---------- | ---------- | --------------- | --------------- | --------------- | ------------ |
| int32        | int[4] | enum[0-11] | enum[0-31] | enum[0-6]       | enum[0 to 6]    | enum[0 to 24]   | varchar(255) |

- **Description:** This Dimension supports the user to explore the types of accident in terms of specific date, week day and time. the time_of_the_day is hourly specific. For example, the user is able to query based on Monday at 8 to 9. Since the keys span over Week day and for each day is 24 hours slot. This allow Fact table to query based time range with sum of the given measures in the Fact Table.

## Weather Dimension

| __Weather_Key__ | Rain_Cond     | Snow_Cond     | X      | Y      | City      | Province |
| --------------- | ------------- | ------------- | ------ | ------ | --------- | -------- |
| int32           | enum[0,1,2,3] | enum[0,1,2,3] | double | double | char[255] | char[2]  |

- **Description:** Weather Dimesnion is gathered from the weather datasets. The X and Y specified the weather station that the data was collected, in addition it allows the weather to connect to the street dimension to enrich the weather condition of the given road.

## Street Dimension

| __Street_key__ | Street_Name | City      | Province | X               | Y               |
| -------------- | ----------- | --------- | -------- | --------------- | --------------- |
| int32          | char[255]   | char[255] | char[2]  | [double,double] | [double,double] |

- **Description:** The motivation for this table is due to the fact the replicated of street name in collision table. Therefore we build a dimension to keep track  one given street. The X and Y coordinates is also a range that used to connect weather data by using the lower and upper bound to compute euclidean distance on each weather station.

  - Further more, the collision location has three different types, collision at intersection , collision at between roads and collision at one given road. X and Y coordinates also play a major role in this. The Range value allow the exploring neighborhoods of one street.

  - If one given street's X and Y coordinates falling in range of another street, this implies these street are connected either intersection or between.

  - This enabled the user to query the collision condition of the neighbors of one given street.

## Collison Specific Table

| __Date_key__ | __Weather_Key__ | __Street_key__ | __Accident_Type_Key__ | nums_fatal_acc | nums_non_fatal_injury | nums_of_accident | nums_of_intersect_accident | nums_of_btwn_accident | total_of_accident |
| ------------ | --------------- | -------------- | --------------------- | -------------- | --------------------- | ---------------- | -------------------------- | --------------------- | ----------------- |
| int32        | int32           | int32          | int32                 | int32          | int32                 | int32            | int32                      | int32                 | int32             |

- **Description:**