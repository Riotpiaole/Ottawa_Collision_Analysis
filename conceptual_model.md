# Conceptual Model

- the impact of severe weather on the frequencies and severities of accidents.

- This goal is to identify neighborhoods, streets and intersections where accidents often occur

- trends in types and frequencies of accidents over  the  four  years,  notably  the  trends  in  numbers of fatal injuries.

## Accident Dimension

| __Accident_key__ | X      | Y      | Is_Fatal  | **Street_keys** | Is_Intersection | **Date_key** | Day_key |
| ---------------- | ------ | ------ | --------- | --------------- | --------------- | ------------ | ------- |
| int32            | double | double | enum[0,1] | int[]           | enum[0,1]       | int32        | int32   |

- **Description:**

## Date Dimension

| __Date_key__ | Year   | Month        | Day          | Week_of_Day_Key | Day_of_the_week | Time_of_the_day |
| ------------ | ------ | ------------ | ------------ | ------------------- | --------------- | --------------- |
| int32        | int[4] | enum[0 - 11] | enum[0 - 31] | enum[0 - 6 ]        | enum[0 to 6]    | enum[0 to 24 ]  |

- **Description:**

## Weather Dimension

| __Weather_Key__ | Rain_Cond     | Snow_Cond     |X      | Y      | City      | Province |
| --------------- | ------------- | ------------- |------ | ------ | --------- | -------- |
| int32           | enum[0,1,2,3] | enum[0,1,2,3] |double | double | char[255] | char[2]  |

- **Description:**

## Location Dimension

| __Street_key__ | Street_Name | City      | Province |
| -------------- | ----------- | --------- | -------- |
| int32          | char[255]   | char[255] | char[2]  |

- **Description:**

## Collison Specific Table

| __Date_key__ | __Weather_Key__ | __Accident_Key__ | __Street_key__ | nums_fatal_acc | nums_non_fatal_injury | nums_of_accident |
| ------------ | --------------- | ---------------- | -------------- | -------------- | --------------------- | ---------------- |
| int32        | int32           | int32            | int32          | int32          | int32                 | int32            |