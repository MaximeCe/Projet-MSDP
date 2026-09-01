2024-02

**MSDP SOFTWARE GUIDE**

Some data processing methods

Pierre Mein

Many processing methods were used until 2023 to process data from MSDP
imaging spectrometers in Meudon Solar Tower, Pic du Midi Turret Dome,
Wroclaw Coronagraph, Tenerife VTT and THEMIS. Except for Wroclaw
observations, processed with IDL software (P. Rudawy, JOSO 1995), all
other data were processed with a Fortran code using the **computation
methods** detailed in this document. We think that most of them can be
used again in modern softwares for new instruments. We note also that,
to increase the accuracy, the methods include some **corrections of
slight distorsions** due to the double pass optics.

Each file from a given sequence corresponds to simultaneous 3D data
(X,Y, Lambda).

Filenames include :

sequence number

date,

time,

and one of the letters  x dark current sequences

y flat field, and possibly field-stop geometry if the spectral line is
not too dark and allows to detect edges of channels

z field-stop geometry (if not obtained with y)

b target observations for scientific analysis

(example in Fig. 1).

<img
src="docs/media/Pictures/100000000000030000000200FAE7C43F5C50D6D4.jpg"
style="width:9.962cm;height:6.613cm" />

****Figure 1. Nine channels of a Meudon MSDP spectro-image in the
H<sub>a</sub> line.

As an example, we shall list now computations of successive steps in the
data processing of a N=9 channels MSDP from Meudon Solar Tower.

For each computation, some parameters from the Fortran codes used in
this example are shown in italics (explanations in the file *msdp.par*
attached after the present document).

Next figures show results obtained for the ground-space paper :

*Bidirectional Reconnection Outflows in an Active Region*

**Ruan,G.; Schmieder,B.; Masson,S.; Mein,P.; Mein,N.; Aulanier,G.;
Chen,Y.

The Astrophysical Journal, Volume 883, Issue 1, article id. 52, 17 pp.
(2019)

Step 1) Averages of sequences for dark current (x), flat field (y) and
field stop (z)

****To get a good accuracy with calibrations, sequences x,y,z must
include many files across quiet regions of the Sun. They must be
averaged.

**Averages** of **dark currents** are subtracted from averages of **flat
field** and **field-stop**, and are used later for all individual files
of target sequences (b).

Step 2) Channel geometry

****It must be noted that coordinates i,j of next figures correspond to
pixels j,i of the detector (to keep the *i* coordinate for the long edge
of field-stop). They will be converted by step 2 into final solar
coordinates (X,Y) in each channel, with X also in the long edge of field
of view.

<img
src="docs/media/Pictures/1000000000000924000006745B0C13A68A753779.jpg"
style="width:16.97cm;height:11.972cm" />

****Figure 2 – Geometry of channels (plot *geo.ps*) :

*Bottom left* : edges of channels , *j* = close to spectrograph
dispersion

*Top left *: Typical intensities (sj) and intensity gradients (*sgj)*
along *j* for detection of channel edges

*Bottom right *: intensities (si) and gradients (sgi) along i

*Top right *: Fluctuations of dimensions obtained for segments
DE,AB,DF,… (see Fig. 3) in

the 9 channels (pixel units)

Figure 2 (top left and bottom right) shows sections of the averaged flat
field file and intensity gradients along i and j , leading to the choice
of **parameters** ***si,sj,sgi,sgj*** in order to determine the accurate
location of channel edges. The corresponding *geo.ps* is plotted
immediately if *igeops*=1.

Variations between successive **channel sizes** (top right) due to the
spectrograph optics are always less than 1 pixel. They show the good
**accuracy** of the results expected from the following detection (Fig.
3).

<img
src="docs/media/Pictures/10000000000000FA000002D578F8F1BA2C991621.png"
style="width:2.782cm;height:7.514cm" />

*****X*

Y

Figure 3. Determination of the XY field-stop solar geometry inside each
channel.

The pixel size, determined by input parameters, is here 0.5x0.5 arcsec.

The accurate location of channel edges is obtained (with smoothing shown
by full lines) through **highest intensity gradients** around points
*a,b,c,d,e,f* (versus *j*) and *k,l,m,n* (versus *i*).

Points ABCDEF (Fig. 3) are **extrapolated** from *a,b,c,d,e,f* and
*k,l,m,n*. ABC and DEF can be parabolic curves because of distorsion (if
parameter *distor=1*).

A short **iteration** is used for the angle of *ac* and *df* versus *i
(*parameter *milangi).*

Inside each channel, intensities are **linearly interpolated** between
left and right channel edges.

The **numbers of pixels** along AC and AD (and DF and CF) depend on
length and width of the field-stop, expressed in arcsec and divided by
the proposed pixel size *milsec/1000.* They define **X and Y
coordinates** of the solar target.

Step 3) Wavelength and intensity calibration

a\) Wavelength translation in successive channels

Several methods can be used to specify the **translation** between
points of the **same wavelength in** successive channels (see Fig. 4 A).
We shall name it **Ts** for **spectrum translation** expressed in Y
pixels.

\- 1) if values of translations between beams of successive channels
going **into** the slicer (**t1**) and going **out** **of** the slicer
(**t2**) are available (here 2.5 and 9 mm), we can use the following
relationship to deduce Ts from the the **geometrical translation**
(**Tg**) of channels *n* and *n+1*

>  **Ts / Tg = t1 / t2**

> where **Tg** is the distance in the output focus between successive
> channels. But Tg must be also expressed in Y pixels, different from
> CCD ones. Let us name **Tg,ij** this distance expressed in the CCD
> coordinates of step (2).

> If **W** is the width of the field-stop expressed in Y pixels (W=
> *ymax-1* if y*max* is the Y dimension proposed by parameter *lj*) and
> **Wij** the same width computed in step (2) with CCD pixels, the ratio
> W/Wij can also define the local distorsion, so that

>  Tg = Tg,ij \* (W/Wij)

> on the condition that Tg,ij and Wij are determined around the same
> channels. This can be done for example by computing Wij in the channel
> 5 (averages between segments AD, CF) and Tg,ij between channels 4 and
> 6 (averages of distances between A,C,D,F in both channels).

> 

> \- 2) a generally less accurate method can use locations of **line
> centers** (smallest intensities obtained by least squares
> approximation) in successive channels (see Fig. 4 A).

\- 3) if the first method is not available, departures between
intensities for several Ts **values **can be compared so that intensity
**departures** between successive channels at the same wavelengths can
be **reduced** to the smallest possible level (see intensity calibration
and Fig. 4B).

> 

> <img
> src="docs/media/Pictures/1000000000000B22000008D21DE487C2F6AB8319.jpg"
> style="width:17cm;height:13.469cm" />

Fig. 4. Calibrations  (plot *cal.ps*) :

A) Line center in channnels 5,6 with respect to coordinates *X,Y*
(pixels 0.5 arcsec)

B) Determination of mean line profile (dashed lines before corrections)

C) Calibration function versus j for two i-values (*imarg* given in
parameters)

> 

> 

>  The Fig. 4(A) shows **line centers** obtained in successive channels
> *ncurv1,ncurv2* (minimum values of parabolic approximation along *Y*).
> In this example, for channel 6, the line center is probably a little
> too close to the edge to give a very high accuracy. The **angle**
> (versus X*)* is deduced from channel 5. The **distances** between line
> centers in successive channels are deduced from the formula **Ts / Tg
> = t1 / t2**.

>  **b) Intensity calibration**

>  **** Constant wavelength versus *X* can be found in each channel
> along line L1 L2 for channel *n* and L3 L4 for channel *n+1* (first
> plot B of Fig. 4), if the distance is *Ts* between both lines.

A full **mean profile** of the flat field spectral line (second plot B
of Fig. 4) can be obtained by comparing parts of the profile deduced
successively from all channels between both lines L1 L2 and L3 L4 (with
the middle point between both lines located in the center of
field-stop), and averaged along a central part of the X field of view.

We introduce also **correcting intensity factors** in successive
channels by adjusting successively averaged intensities of the same
wavelengths defined by lines L1 L2 and L3 L4 in all couples of channels
*n* and *n+1*. Little departures can be seen between mean profile curves
printed before correction (**dash** lines) and after correction
(**full** line).

c\) Calibration function

******A **calibration function** *fl(X,Y,n)* is defined as the flat
field intensity divided by the mean profile intensity in the same
wavelength (taking into account the wavelength function along the
*X*-coordinate).

In the following steps, **calibrated intensities** wil be obtained by
dividing the observed intensities by the calibration function in each
point X,Y*,n.*

Step 4) Computation of c-files → calibrated channels

By subtracting averages of dark currents and by using the geometry and
the calibration function derived from flat fields, we can now, for any
file of a time-sequence, compute a *c-file* giving an i**ntensity map**
of the N observed channels (Fig. 5).

<img
src="docs/media/Pictures/10000000000000DD000001142D94FA461AB1576B.jpg"
style="width:5.805cm;height:7.26cm" />

***** Y*

***** X*

******Fig. 5 Calibrated intensity map of the N channels (c-file).

Step 5) Computation of d-files → line profiles, filtergrams and
Intensity/Velocity maps (bisector method)

In each solar point, the **line profile** of N=9 points can be
interpolated at first into 4xN-3=33 wavelength points, by third degree
**interpolation** (profiles in *cmd* files)*.* Each small interval is
then interpolated linearly. To get **intensity map at a given
wavelength** similar to a filtergram**,** a new local interpolation can
be used. To compute **intensity and velocity** corresponding to a given
distance between two points of equal intensity (**bisector** method), a
linear interpolation is used again in the profile of 4xN-3 points. It
can be noted that dopplershifts are generally determined in parts of
profiles near inflexion points, reducing interpolation errors. ****

The previous **polynomial interpolation** may introduce errors in
intensity and velocity maps. Because dopplershits are small as compared
to wavelength distance between channels, errors are similar along large
X distances for a given Y value. For **solar disk** observations with
full solar fields of view, it is possible to reduce such departures
(parameter *cordisk=1).* For each Y value, a parabolic approximation is
computed along X. Errors are roughly periodic due to successive useful
channels along Y. Then for each (X,Y) value, intensities and velocities
can be corrected by subtraction of the difference between approximated
X-parabolas and averaged values over Y to **reduce possible errors.**
Residual mean zero velocity, due partially to the non-integral number of
Y periods in the field of view, can be corrected by the *mcorrec*
parameter (see step 7). ****For prominences, the field of view is not
full enough of solar structures for that. But It can be noted that, for
prominences, velocities are generally larger, and a lower accuracy is
more acceptable. ****

*Step 6) Severald-files → Scans of full targets → q-files*

When **scans** are used to observe wide targets, several parameters
define the files to be used simultaneously to compute XY maps (*nob1,
nob2, ntmax, priscan*). **2D correlations** between common fields of
successive images are used to associate successive d-files (parameter
*lcorrel* chooses generally line center maps). Correlations values are
listed in *scan.lis*.

Resulting filtergrams and intensity/velocity maps are recorded in
**q-files**.

***Step 7) Plots of q-files maps***

The code uses the standard subroutine PGGRAY to plot quickly maps for
each intensity or velocity map. Parameters *blackq* and *whiteq* control
the plots of q\* maps. They define **level**s corresponding to **black**
and **white** colors. The *scan.lis* file gives the **averaged
velocity** corresponding to the map of the first Velocity map. This
averaged velocity refers to the channel *ncurv1*. A parameter
(*mcorrec)* is available to **modify the zero value** in velocity maps.

<img
src="docs/media/Pictures/10000000000003520000027CBFF7341C0F9379A7.jpg"
style="width:10.758cm;height:8.001cm" />

Fig. 6 Filtergram at line center.

<img
src="docs/media/Pictures/100000000000035200000538C8020082B1FBBB96.jpg"
style="width:10.758cm;height:16.93cm" />

******Fig. 7. Maps for intensity (top),and velocities (bottom) from
bisector method at ± 45 pm*.*

Figures 6 and 7 show the line center intensity and the couple
Intensity/Velocity with bisector method at ± 45 pm, with the option of
**improved interpolation** (*cordisk=1*). For velocities, black and
white correspond to -3 and +3 km/s in this example.

Note : New Spectro-imaging Polarimetric observations with MSDP

New polarimetric observations have been tested with the Meudon S4I
prototype. They ave been details in the papers

Four decades of advances from MSDP to S4I and SLED imaging spectrometers

**Mein P., Malherbe J-M., Sayède F., Rudawy P., Kenneth P., Keenan F.,

2021, *Solar Physics,* Volume 296, Issue 2, article id.30,

arxiv:2101.03918

*Optical capabilities of the Multichannel Subtractive Double Pass (MSDP)
for imaging spectroscopy and polarimetry at the Meudon Solar Tower*

Malherbe,J-M., Mein, P., Sayède,F.,

2023arXiv231202555M

The width of the field-stop is divided by 2. A simple birefringent plate
creates 2 parallel beams with 2 linear orthogonal polarizations I+S,
I-S. As a result, twice as many channels (18→36 with S4I) are obtained
simultaneously with both polarizations. Such methods could be used with
new MSDP spectrographs, and with processing steps similar to the steps
listed in this report.

##      **                                                                        

file msdp.par

========

Parameters for MSDP observations

(format a8,i8)

name, value

(several lines if dimension\>1)

\* parameters often modified

Meudon Solar Tower 2017-03-30 10:20:03 (solar disk)

SUCCESSIVE STEPS

================

ixy 1 \* computation of average files dark, flat, field-stop

igeo 1 \* geometry

iflat 1 \* calibration

ibmc 1 elementary calibrated c-files (1 file per time)

icmd 1 d-files (spectroheliograms and I/V maps)

iquick 1 q-filesfor full scanned targets

igrayq 1 plots of q-files

idem

--------

igeops 1 automatic plot of geo.ps

icalps 1 automatic plot of cal.ps

ndprofps 3 number of d-file for automatic plot of profiles 3

see Plots of Line Profiles for parameters

igrayqps 1 automatic plot of sq....ps

SELECTED OBSERVATIONS

=====================

nob1 1 first image to be processed 1

nob2 5 last image to be processed 300

ntmax 5 number of images per scan

priscan 4 = 4 for Meudon (5 images), scanning by prisms (2,3,1,5,4))

nobstep 5 step between first images of two scans

dob20170330 date of observation (optional)

tob110200300 \* first time for observations to be processed

tob224000000 last time

tdc1 0 09564585 same for dark current

tdc224000000 09564585

tfs109593900 09571701 same for field stop

tfs224000000 09572084 09571701

calfs -1 0

-1 if flat field Y replaces field stop Z

tff109593900 09571701 same for flat field

tff224000000 09572084 09571701

nff 1 number of flat fields

------

GEOMETRY

========

li 442000 field stop length (arcsec/1000)

lj 61000 field stop width

jypas 45000 \< translation between images (for correlations)

nline 1 spectral line index

ncam1 1 detector index

nm 9 number of channels

lbda 6563 line wavelength (Angst)

dlbd 300 wavelength distance between channels (mA)

mupris 9000 translation between output channels (microns)

mustep 2500 distance betwwen successive slits (microns)

nwinp 1 number of simultaneous detectors

interc 15 approximate distance between right edge of a channel

and left edge of the next one (unit = CCD pixel)

nbcln 1024 number of X-CCD pixels

nblgn 1536 number of Y-CCD pixels

invern 0 0 j-value of pixels of same wavelength

decreases with channels

1 j-value of pixels of same wavelength

increases with channels

idc 1 generally 1

0 = dark current not subtracted

-1 = dark current not abvailable

si 15 intensity threshold for channel edges detection versus X

0 = automatic detection for si,sgi,sj,sgj

sgi 10 intensity threshold for intensity gradients versus X

sj 15 the same for intensity versus Y

sgj 10 the same for gradients versus Y

milangi -40 approximated angle between longer edge of channels

and CCD (radian/1000)

milgeo 2000 threshohld for geometry accuracy:,maximum departure between

values and regression lines (plot geo.ps, unit pixel/1000)

nleft 0 interpolated approximations for bad channels (left)

nright 0 the same (right)

i1 1 first useful pixel in the i-direction

i2m 0 the last useful pixel in the i-direction is

i2=im-i2m (im = total number of pixels)

j1 1 same definitions for j

j2m 0 .....

lip 40 the curvature of channels is determined by

3 intervals around 3 points of the longer edge.

If L is the length of this edge, the points

are located at

L\*(o.5-lip/100)

L\*(o.5+lip/100)

jeps 20 The accurate determination of the longest edges

is searched around

approximate values +/- jeps pixels

intvi 60 The edges parallel to i are determined by cuts

along j, averaged over the interval

+/- intvi around the 3 points metioned above

(see lip)

intvj 30 Similar definition for short edges parallel to j,

but with the intervals

left end - left end +intvj

right end - right end - intvj

leps 50 The detection of points with maximum gradient

is made in 2 steps:

 - approximate values corresponding to

signal = intensity threshold

 - search of maximum gradient in intervals

+/- leps around these values

n1 1 The useful channels are numbers n so that

n1.le.n.le.(nm-n1+1)

where nm is the total number of observed channels

distor 1 1 = curvature of channels taken into account

for detection (check not included in calculation)

0 = curvature not taken into account

calfs -1 -1 = field stop is replaced by flat field

for geometry

-----

iswap 1 1 for LINUX

ipermu 1 permutation of CCD X and Y coordinates

milsec 500 output pixel (unit arcsec/1000)

CALIBRATION

===========

(msdp2.f: cmf1.f -\> cal.ps, bmc1.f -\> fichiers c\*)

icalct 0 0 = calibration by flat field

1 = calibration by field stop (continuum)

inclin 1 0 = line profiles determined by previous calculations

(-\> jt1000,ja1000,jb1000)

1 = determination of mean curve of absorption line

ncurv1 5 first channel plotted for line profile determination (4)

ncurv2 6 last channel ..... (6)

il1p 1 beginning of computation interval of meanline profile (%)

il2p 99 end .....

curv 0 1 = line curvature taken into account

0 = inclination of line only taken into account

iliss 60 smoothing over 2\*iliss+1 i-points before detection of

line center

jparli 5 parabolic smoothing over 2\*jparli+1 j-points for

line center

lispro 5 parabolic smoothing over 2\*lispro points of

mean line profile (-\> fl profile)

imarg 200 margin for small and large X (i0=1+imarg, i0=im-imarg)

for fig. C in cal.ps

FILTERGRAMS + MAPS INTENSITY/VELOCITY

=====================================

(msdp3.f: cmd1.f -\> files d\*)

SUPPRESSION of a few arcsec near EDGES

NEW RESTRICTED FIELDS

(msdp3.f: cmd1.f -\> d\* files)

ix1 1000 if ix1\>0 and/or ix2\<li, the length of the field of view

is restricted to ix1\<--\>ix2 (unit arcsec/1000)

This allows to eliminate bad points at the edges of the

field-of-view (1000 = 2 arcsec suppressed)

No suppression means ix1=0, ix2=li, jy1=0, hy2=lj

ix2 441000

jy1 1000 similarly, new limits of the width of field-of-view

(jy1\>0 and jy2\<lj).

The difference jy2-jy1 must be larger than the step "jypas"

to save some overlap between successive images of each scan

jy2 60000

jyq1 2000 similarly, limits of the field-of-view for "d" and "q"

files (jyq1\>jy1 and jyq2\<jy2), in the same coordinates as

for jy1 and jy2

(if inverj=1, the code converts automatically

jyq1 into jy2-jyq2

jyq2 into jy2-jyq1)

If zero, automatically put to jy1 and jy2.

jyq2 59000

cented 1 1 = line center map

=== number of filtergrams/sums and diff/IV couples ===\|

lmpd 2 filtergrams for lmpd lambdas

0 the same for sums and differences of filtergrams (not used)

1 \* number of couples Int/Velocity with bissector

ex.: cented+2\*lmpd = 1+2\*8=17 points in the profile

(1 = 2 maps, 2 = 4 maps, ...)

=== First dlambda for filtergrams and I/V couples ===

(unit dlbd/1000 :

for Meudon 1000 means 300 mA = wavelength between channels)

lbd1d 1000 first dlambda for filtergrams (unit dlbd/1000)

0 first dlambda for sums and differences (not used)

1500 \* first dlambda for couples I/V 1500

(3 lines)

=== dlambda steps ===

lbpasd 1000 step for dlambdas of filtergrams (unit: dlbd/1000)

0 step for sums and differences (not used)

0 step for bissector I/V maps

(3 lines)

ispline3 1 1 = profile interpolation degree 3

0 = profile interpolation degree 4

inveri 1 1 to invertX-values in the d maps (0 = no inversion)

inverj 0 the same for Y-values

inverl 1 1 to invert the wavelength orientation in the spectrum

mps 1 velocity unit = mps m/s

norma 0 Normalization for thin clouds correction. No effect if = 0.

norma = 'abcd' with a,b,c,d integers of one figure:

let us call L the total length of the field.

Normalizarion is calculated over 2 intervals

ab -\> from 0.1\*L\*a to 0.1\*L\*b

cd -\> c d

Both intervals should avoid active regions

If d=0, d is replaced by 10

example: 0059: one interval 0.5L to L

PLOTS OF LINE PROFILES

======================

icplot 440 center point for plots (index c-files: pixels)

jcplot 60 " " "

icstep 220 i-step

jcstep 30 j-step

njm 3 number of lines and columns

POSSIBLE LINE PROFILE INTERPOLATION CORRECTIONS FOR DISK

cordisk 1 correction of disk line profile interpolations,

by equalization of mean values along X.

FOR PROMINENCES (iminpro.ne.0)

===============

iminpro 0 minimum intensity for prominences (bissector)

nodisbis 0 1 = no plot disk(absorpt.line) for bissector map (1442msdp3

FOR SCATTERED LIGHT IN PROMINENCES

==================================

ilin1 0 X-value (arcsec) pour j=1 (0= no scattered lght correction)

0

0

0

0

ilinm 0 X-value for j=jm

0

0

0

0

linvisu 0 intensity to visualize vectors of scattering

intadd 0 additional intensity for plot of weak prominences

maxwing 0 no linscat correction for wing intensity \> maxwing (disk)

FULL TARGETS

==============

(dme1.f -\> fichiers q\*)

invi 0 1 to invert X in the output maps of q-files

invj 0 1 to invert Y in the output maps of q-files

CORRELATIONS BETWEEN SUCCESSIVE IMAGES

lcorrel 0 filtergram number in d-file (generally line center)

used to correlate successive parts of scan.

Note that in d-file the filtergrams are in order

left wing / center / right wing

if = 0: autonatically lcorrel = map intensity line center

copasq 4000 step for correlations unit arcsec/1000

(\< structures: mottles..)

milcoq 100 correlations accepted if corm \> milcoq/1000 (see scan.lis) ?

POSSIBLE CORRECTION OF ZERO DOPPLER VELOCITY

vrejec 10000 rejected velocities for mean velocities (m/s)

mcorrec 119 \* correction of reference velocities (m/s)

(mcornew red in previous scan.lis)

for prominence velocities and mcorrec calculation:

minint 0 min I for mean velocities (no effect if 0)

maxint 10000 max I ................... (no effect if 10000)

minmax 0 minmax=1 -\> V=0 for I\<minint or I\>maxint

PLOTS OF Q FILES

================

(grayq msdp4.f gray1 subgray)

vps 1 1 = vps

igrq 1 nombre de plots disposes horizontal.t dans la page sq...ps

jgrq 3 ........................ verticalement .................

slw 2 line width

milsch 1500 character size \*1000

smallq 0 if =1 takes into account iqa,iqb,jqa,jqb for small plot

iqa 0 starting point X (arcsec) in field of q-file plot

iqb 0 end point X

jqa 0 starting point Y

jqb 0 end poiny Y

blackq 0 minimum values

0

350

0

0 5

600

-3000

0

0

0 10

0

0

0

0

0 15

0

0

0

0

0 20

whiteq 0 maximum values 1 100

0

600

0

0 5

1000

3000

0

0

0 10 line center

0

0

0

0

0 15

0

0

0

0

0 20

end
