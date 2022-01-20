from datetime import datetime

given_date = "April 24, 2022, 10:33 p.m.";
d = datetime.strptime(given_date, "%d-%b-%Y-%H:%M:%S")

print(d.strftime("%Y-%m-%d-%H:%M:%S"))


#2017-11-01 12:00:13
#2022-04-24 22:33:19.000000
