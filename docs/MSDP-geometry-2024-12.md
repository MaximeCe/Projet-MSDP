2024-12

MSDP SOFTWARE GUIDE

*Some remarks about geometry of MSDP channel*s

Pierre Mein

We present some additional remarks to « processing methods » of MSDP
data (2024-02).

They correpond to last versions of Fortran codes, in the step of
« **Channel geometry **».

 1) The upper part of Figure 1 (file *geo1.ps*) shows the cross-sections
of *intensity* and *intensity gradient* along the central part of field
of view plotted in the lower part (section y=500). Maximum values are
adjusted to 100 in both cases.

<img
src="docs/media/Pictures/10000000000002EB000002FBB34428BB707AC49D.jpg"
style="width:0.042cm;height:20.188cm" /> The lower part is the full
field of view with 3 sections at Y-values 150, 500, 850, used to
determine the left and right edges of all channels, as seen in Figure 3.

The full drawing of channel edges is shown also, according to all
determinations using cross-sections X and Y described in Figure 3 .

<img
src="docs/media/Pictures/10000000000002EB000002FBB34428BB707AC49D.jpg"
style="width:0.042cm;height:20.188cm" /><img
src="docs/media/Pictures/1000000000000249000002C08E639443DE7FDF5E.jpg"
style="width:9.881cm;height:12.095cm" />

Figure 1

The ***intensity gradients**,* shown in Figure 1, show that, in spite of
low intensity of the spectral line, gradients can be used to determine
channel edges, on the condition that their absolute value exceeds some
**thresholds**. Quantities *******mgx*** (minimum gradient for left side
of channels) and **-*****mgx*** (right side) are selected through Figure
1.

We can note that, to increase signal-to-noise ratios, because of the
**very small angle** between sides of channels and Y axis (around 0.04
rad), the detection of edges can be done also with **average intensity
gradients** of some horizontal sections, around each of the 3 above
mentionned.. The parameter ***ladd****x*** (additional lines X, up and
down) leads to average gradients between Y-laddx and Y+laddx.

In the example of our MSDP observation, 25 flat-field data are averaged.
The used *laddx* is 0. In the case of observations with far fewer data,
*laddx* could be accepeted easily up to 5 (shifts ± 0.2 pixel in extreme
sections).

2) Figure (2) is an enlargement of the upper part of Figure 1. The
cross-sections of *intensity* and *intensity gradient* are plotted again
along the central part of field of view (Y=500), but only 120 pixels
around the left edge of first channel.

The *maximum gradient* (greater than **mgx**) is plotted between
**intensities** X=97 and X=98 (upper part of the figure), and in
**intensity gradient** X=97.5 (lower part) .The **parabolic
interpolation** of intensity gradients is used with the 3 gradients in
X= 96.5, X=97.5 and X=98.5 (lower part). The **parabola** is printed
with a thick line.

Gradients X=97.5 and X=98.5 are in this example almost equal. So the
maximum of interpolated parabola, defining the first left channel edge,
is close to the middle abscissa X=98. In Figure 3, this will define the
point **b**, which becomes **B** in final drawing of channels.

<img
src="docs/media/Pictures/10000000000002490000025052CF0C00AF28BC0F.jpg"
style="width:13.099cm;height:13.256cm" />

Figure 2

3) Figure 3 (geo3.ps) shows the **first** **channel geometry** :

 - points **a,b,c,d,e,f** determined by using the 3 cross-sections of
Figure 1 with the same method as seen in Figure 2

\- points **k,m,l,n** by using Y cross-sections, starting from Y values
of **ad** and **cf**, and X-values shifted by 25 pixels towards channel
inside. Unlike X cross-sections used with a,b,c,d,e,f, each Y
cross-section looks here for only one point. A new gradient threshold
**mgy** is available (here the same as *mgx*).

<img
src="docs/media/Pictures/100000000000010D000002E7F7A16892FE5C8CF7.jpg"
style="width:5.844cm;height:16.138cm" />

Fiure 3

It can be noted that the coordinates of points a,b,c and d,e,f of Figure
3 can be used to estimate the **distortion** due to spectrograph. In
this example, the quantities for all channels

X(b)-(X(a)+X(c))/2 (left side) and X(e)-(X(d)+X(f))/2 (right side)

have respective averages - 0.08 and + 0.19 (with the pixel unit close to
0.5 arcsec), with departures less than ± 0.2.

This is very small and suggests to discuss again in this case the option
of parabolas, adjusted to channel sides **abc** and **def.**

4) Figure 4 (geo4.ps) gives the lengths in X and Y of the sides of 9
channels, AC, DF, AD and CF. The very small departures between
successive channels show the good **accuracy of the parabolic
interpolation**.

<img
src="docs/media/Pictures/100000000000022D000002A7695DB77A43E47D6F.jpg"
style="width:14.148cm;height:17.247cm" />

Figure 4
