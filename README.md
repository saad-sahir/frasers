google drive: https://drive.google.com/drive/u/0/folders/1MqmELp7vAIS01KLqw0yUcAqT_sP12w49

- Car parked at drop off for too long (revsc1)
- Persons loitering at location (revsc2)
- Private car parked at loading bay (revsc3)
- Detect carpark gantry jam (revsc1)
- Detect jam inside car park (N/A)
- Turnstiles jam (revsc2)
- Activites at open area after hours (revsc4)

+ revsc1: 
------------
this script focuses on detecting cars that have been in an area 
for too long. in the context of carpark gantry, if a car has been
stuck in front of the gantry for too long, one can assume that there
is a jam in front of the gantry

+ revsc2
------------
this script focuses on detecting people that have been in an area for
too long. thus, people loitering in an area and in the context of the
turnstiles, one can assume that there is a jam in front of the turn-
stiles

+revsc3
------------
this script focuses on detecting cars in an area where they shouldn't
be (such as the loading bay for trucks)

+revsc4
------------
this script focuses on detecting people after a certain time of the
day to alert security of intruders, thus "nightwatch".
