c     ms2.f
c========
c                                        bmg1.f, bmc1.f, cmf1.f
c**************************
c  Programme geom                        bmg1 - UNIX
c*************************************************************************
      subroutine geom(nw,win,nm,iux,iuy,iuz,gname,istop,ima,ijcam,
c                                                       out
     1     imima,jmima,xr,yr,imc,jmc,iim,jjm)
c     out............               885  123
      


c       linc     CCD        imb    1536     jmb   1024
c     permut     imc    1024     jmc   1536
c     channels   imd     885     jmd    123     nm   9

      
c       character comm*80,fin*3,end*3
       character*22 gname
       dimension kz(512)
       integer si,sgi,sj,sgj,distor,win(8)
c     integer winp,sgj1,sgj2
       integer*2 imadark(1536,1536)
       integer*4 ima(1536,1536) ! ,zmi(2000,3)
c     attention images < 1536*1536
       dimension xr(24,3,2),yr(24,3,2)

c--------------------
c                       voir SRECT
c      is=1536
c      js=1536
c     leps    ! intervalle max entre point probable (seuil) et gradient max
c      milanglei=60
c        anglei=milanglei*0.001
c      idangle=0.2*(i2-i1)
c              demi-disrtance entre les 2 coupes utilisees pour estimer 
c                                                          l'angle/i   
c                  coupes a ic+/-idangle, integration sur +/-intvi
c     lip         pourcentage ecarts en I pour parabole haut/bas (bords J)
c          ex: 30
c      intercan=6  ! nombre de pixels entre canaux <---------- modifiable
c     icp       inutilise:
c         coupes //j autour de ic=(i1+i2)/2
c     jeps        pour la determination precise des points en J,
c               recherche +/-jeps  autour des valeurs approchees
c     intvi  intervalle integration en i pour detection bords j
c     intvj                            j                      i
c--------------------------------------
       print *,'debut geom: nw,gname',nw,gname
       write(3,*)'debut geom: nw,gname',nw,gname
1      format(a)
2      format(1x,a)
3      format(12i6)
4      format(1x,12i6)
5      format(//)
6      format(3a4,3i8)
7      format(2x,3a4,3i8)

       call par1('    igeo',nw,igeo)
c       if(igeo.eq.0)then
c          stop
c          return
c       endif       
c      call par1('   nbcln',nw,nbcln)
c            coef=float(nbcln)/1000.
      call par1('  interc',1,interc)
c                                     interc=interc*coef
      call par1('      si',nw,si)
      call par1('      sj',nw,sj)
      call par1('     sgi',nw,sgi)
      call par1('     sgj',nw,sgj)
      call par1(' milangi',nw,milangi)
      call par1(' milangj',nw,milangj)
      call par1('  milgeo',nw,milgeo)

      call par1('      i1',nw,i1)
      call par1('     i2m',nw,i2m)
      call par1('      j1',nw,j1)
      call par1('     j2m',nw,j2m)
      call par1('     lip',nw,lip)
c                                    lip=lip*coef
      call par1('    jeps',nw,jeps)
c                                    jeps=jeps*coef
      call par1('   intvi',nw,intvi)
c                                    intvi=intvi*coef
      call par1('   intvj',nw,intvj)
c                                    intvj=intvj*coef
      call par1('    leps',nw,leps) 
c                                    leps=leps*coef
      call par1('      n1',nw,n1)
      call par1('  distor',nw,distor)

      call par1('  normsq',nw,normsq)
      call par1('    norm',nw,norm)
      call par1(' largrid',nw,largrid)

       n2=nm-n1+1
       nc=(n1+n2)/2
       n3n4=0
      print *,' read yname '
       rewind(iuy)
       read(iuy)kz
       print 3,(kz(i),i=1,8)
       write(3,*)' geom kz(2),kz(3)  (im,jm)',kz(2),kz(3)   !    3)(kz(i),i=1,8)
       im=kz(2)
       jm=kz(3)
c
      print *,' geom: i1,i2m,im,  j1,j2m,jm ',i1,i2m,im,  j1,j2m,jm
      write(3,*)' geom: i1,i2m,im,  j1,j2m,jm ',i1,i2m,im,  j1,j2m,jm
      print *,' call SRECT'
      call srect(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
c          -----
     1           intvi,intvj,interc,
     2           si,sgi,sj,sgj,leps,
     3           nm,n1,n2,nc,n3n4,distor,
     4           iux,iuy,gname,
     5           ima,imima,jmima,imadark,
     6            milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
           write(3,5)
       return
       end    
c-------------------------------------------------------------------
      subroutine SRECT(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
     1                 intvi,intvj,intercan,
     2                 sip,sgip,sj,sgj,leps,nm,n1,n2,nc,n3n4,
     3                 distor,
     4                 iux,iuy,gname,
     5                 ima,imima,jmima,imadark,
     6                 milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
      integer*2 lec(1536),lecx(1536),meanflat(1536,1024),
     1          imadark(1536,1536)
      integer*4 ima(1536,1536)
      dimension z(1536),zg(1536),x(1536),zmoy(1536),zmoyg(1536),
     1        xr(24,3,2),yr(24,3,2),yc(24,2,2),yca(24,2),
     2        nkbon(24,2),zmi(2000,3),y(1536),
     3        vi(24,6,6),vj(24,6,6),vmod(24,6,6),flech(24,2),ili(24,2),
     4        xdes(7),ydes(7),avi(6,6,3),avj(6,6,3),avmod(6,6,3),
     5       kz(512),zd(1536),xbord(24,5,2),ybord(24,5,2),
     6       integ(2,3),ecarti(24,6,6),ecartj(24,6,6),ecartmod(24,6,6),
c                deb-fin
     7       ksi(20),ksj(20),ksgi(20),ksgj(20)
      integer sip,sgip,si,sgi,sj,sgj,sgj1,sgj2,distor,displ,head(512)
c                                                           int4, tab int2
      character titre*54,gname*22
c      integer*2 listj(1536)
c      dimension cis(3,2),ypar(11,3,2)
c      character nomps(24),gameo(22)

      data ksi /2,2,4,4,4,6,6,6,9,9,9, 12,12,12,15,15,15,20,20,20/
      data ksgi/2,2,4,4,4,6,6,6,9,9,9, 12,12,12,15,15,15,20,20,20/
      data ksj /2,4,2,4,6,4,6,9,6,9,12, 9,12,15,12,15,20,15,20,25/
      data ksgj/2,4,2,4,6,4,6,9,6,9,12, 9,12,15,12,15,20,15,20,25/

      iug=34
      print 3,ksi
      
 3    format(20i3)
c      call system('rm channel.lis')
c      open(unit=95,file='channel.lis',status='new')
      callsystem('rm xryr.lis')
      open(unit=7,file='xryr.lis',status='new')
      
      displ=0

c          indices      xbord (X//i)
c                       ybord (y//j)
c                  (canal n)
c                 
c                                  n,5,1   n,5,2
c                                  --+-------+--
c                                  |           |
c                           n,3,1  +           +  n,3,2
c                                  |           |
c                           n,2,1  +           +  n,2,2
c                                  |           |
c                           n,1,1  +           +  n,1,2
c                                  |           |
c                                  --+-------+--
c   
c                                  n,4,1   n,4,2

c        nsm=21
      nsm=20
      print *,' sip ',sip
        if(sip.ne.0)nsm=1
        ecmin=10000.
        nsol=0
        ipl=0
        ecartbis=1000.
      print *,' ecartbis ',ecartbis
c        iux=31
c        iuy=32

      call par1('     idc',1,idc)
      call par1(' kdangle',1,kdangle)
c        if(kdangle.eq.0)kdangle=200
c                     idangle=kdangle*(i2-i1)/1000.
c=================================================================
      print *,' nsm ',nsm
      write(3,*)' SRECT nsm ',nsm
      do 190 nseuils=1,1         !nsm        !      ----------------------- nseuils
       if(nseuils.eq.nsm)ipl=1
c                        threshold loop
1     format(' geom: nseuils, si,sgi,sj,sgj',i3,2x,4i4)

      if(sip.eq.0)then
c      si=3*nseuils
c        if(nseuils.eq.nsm)si=3*nsol
      si=ksi(nseuils)
         if(nseuils.eq.nsm)si=ksi(nsol)
      sgi=ksgi(nseuils)
         if(nseuils.eq.nsm)sgi=ksgi(nsol)
         if(sgip.lt.0)sgi=-sgi
      sj=ksj(nseuils)
         if(nseuils.eq.nsm)sj=ksj(nsol)
      sgj=ksgj(nseuils)
         if(nseuils.eq.nsm)sgj=ksgj(nsol)

      else
      si=sip
      sgi=sgip
      endif
               print 1,  nseuils,si,sgi,sj,sgj
               write(3,1)nseuils,si,sgi,sj,sgj
      is=ijcam
      js=ijcam
      n3=n3n4/10
      n4=n3n4-10*n3
        if(n3.eq.0)n3=n1
        if(n4.eq.0)n4=n1

c     leps    ! intervalle max entre point probable (seuil) et gradient max

        anglei=milangi*0.001
        anglej=milangj*0.001

c      kdangle=  lu sur ms.par  (idangle=kdangle*(i2-i1)/1000)
c                  coupes a ic+/-idangle, integration sur +/-intvi
c     lip         pourcentage ecarts en I pour parabole haut/bas (bords J)
c      intercan= nombre de pixels entre canaux
c     icp         inutilise


      xa=0.1
      xb=0.6
       
      xc=0.65 !0.7
      xd=0.95 !0.9
      xcd=0.80  !0.5*(xc+xd)

        xcd1=0.75
        xcd2=0.85

        do2 n=1,24 !                             ????????????
c                    10 bords detectes sur 4 cotes (+2 inutilises)
        do kg=1,2
          do ihb=1,5
          xbord(n,ihb,kg)=0.
          ybord(n,ihb,kg)=0.
          enddo
c                     6 points de reference sur 2 cotes
          do ihb=1,3
          xr(n,ihb,kg)=0.
          yr(n,ihb,kg)=0.
          enddo
        enddo
2       continue

      i2=im-i2m
      idangle=float(kdangle)*(i2-i1)/1000.
c                            0.2*(i2-i1)
          write(3,*)'idangle',idangle
      ic=0.5*(i1+i2)+0.5
      centre=0.5*(i1+i2)
      icentre=ic  

      j2=jm-j2m
c      ia=ic-intvi
c      ib=ic+intvi
         iap=ic-intvi
         ibp=ic+intvi
c      write(3,*)' im,jm,i2,ic,j2,ia,ib ',im,jm,i2,ic,j2,ia,ib
c
        do j=1,ijcam
        do ihmb=1,3
        zmi(j,ihmb)=0.
        enddo
        enddo
c
c          den=normsq       !???
c          den=den/10.
c      do4 ilec=1,3
c      den=den*10

        print *,' read iux,iuy'
      rewind(iuy)
      read(iuy)kz
c---------------------          stockage de l'image entiere
       rewind(iux)
       read(iux)kz
       print *,' dark head '
       print *,(kz(i),i=1,3) 
       write(3,*)' dark head 1-8 ',(kz(i),i=1,8)
c       im=kz(2)
c       jm=kz(3)

      iia=i1
      iib=i1+4
      iic=i2-4
      iid=i2

      print *,' subtraction y-x    im,jm ',im,jm
      imd=1024
      jmd=1536
      write(3,*)' mdark iux ',iux
      print *,  ' mdark iux ',iux
c      iux=31
c      call mdark(iux,imadark,imd,jmd)
c     write(3,*)' mdark imadark(1,1) ',imadark(1,1)
      
      imdark=im
      jmdark=jm
      write(3,*)' imdark jmdark ',imdark,jmdark
      imima=im
      jmima=jm
      do j=1,jm
         read(iuy)(lec(i),i=1,im)
         if(j.eq.1)print *,' lec 1,im,100 ',(lec(i),i=1,im,100)
         
        if(idc.eq.1)then                               ! oui idc=1
        read(iux)(lecx(i),i=1,im)
        do i=1,im
c           imadark(i,j)=lecx(i)                    !      mean dark
           lec(i)=lec(i)-lecx(i)                   ! subtraction
           if(lec(i).lt.0)lec(i)=1
c                 piv=abs(lec(i))
c                 lec(i)=sqrt(piv)
         enddo
        if(j.eq.1)print *,' y-x 1,im,100 ',(lec(i),i=1,im,100)
        endif
c      if(j.eq.200)write(3,*)' bmg:lecx ',(lecx(i),i=1,100,10)
c      if(j.eq.200)write(3,*)' geo:lec  ',(lec(i),i=1,100,10)

c      call zero3(iia,iib,iic,iid,0,0,0,lec,1,im,1,0,
c     1          az1,az2,az3,az4,az5,az6,az7,az8,az9)
c                          on retranche une fonction lineaire joignant
c                          les fonds en debut et fin de ligne
c        if(j.eq.200)write(3,*)' apres zero'
c     if(j.eq.200)write(3,*)(lec(i),i=1,286)
        normsq=0
        do i=1,im
c         if(normsq.eq.0)then
          ima(i,j)=lec(i)                              ! ima
c         else
c         print *,'  normsq.ne.0 for i,j ',i,j
c         kpiv=lec(i)
c         ipiv=kpiv*kpiv/den
c           if(ipiv.gt.32000)goto4
c         ima(i,j)=ipiv
c         endif
        enddo
      enddo
        goto5
4     continue
 5    continue
      write(3,*)'ima', ima(1,1),ima(im,jm)
 8    format(i4,2x,10i4)
      do j=1,jm,200
         print 8,j,(ima(i,j),i=1,im,200)
      enddo
c--------------                   ima  test
 9    format(i8,2x,11i5)
         i1t=im*0.1
         i2t=im*0.9

      write(3,*)' ima test   im,jm  i1t,i2t   ',im,jm,' ',i1t,i2t
c       write (3,9)i1t,(ima(i1t,j),j=75,125,10)  
c       write (3,9)i2t,(ima(i2t,j),j=75,125,10)    
c============================================      
 190  continue
      
 200  continue
c                calcul de imaperm puis meanflatmd   (meanflat minus dark)
      do j=1,1024
         do i=1,1536
            meanflat(i,j)=ima(j,i)
         enddo
      enddo
c-------------------------
      call newgeom(meanflat,xr,yr)    !   New geometry
      
         return
         end
c=============================================
      subroutine SMAX(z,i,eps,apar,bpar,cpar)
      dimension z(1536)
c      eps=0.5
c      call par1('  interp',1,interp)
c     parabolic interpolation of intensity gradients
c      write(3,*)' smax: interp eps',interp,eps
c      if(interp.eq.1)then 
      bpar=(z(i+1)-z(i-1))/2.           
      apar=(z(i+1)+z(i-1))/2.-z(i)  
      cpar=z(i)
        if(apar.eq.0.)return
      eps=-bpar/(2.*apar)
c      endif
     
c      b=z(i+1)-z(i-1)           !  /2 
c      a=z(i+1)+z(i-1)-2.*z(i)   !  /2 
c        if(a.eq.0.)return
c        eps=-b/(2.*a)
c 1      format(' SMAX:   imax,     z(i-1),z(i),z(i+1),          eps ')
c 2      format(8x,        i5,   5x,3f6.0,                7x,f8.2)
c     write(3,1)
c     write(3,2) i,z(i-1),z(i),z(i+1),eps 
      return
      end
c===========================================
c      subroutine newgeom(z,i1,i2,zseuil,zgseuil) !,iedges)
      subroutine newgeom(meanflat,xr,yr)
      integer*2 meanflat(1536,1024)
      dimension ja(3),sig(2)
      dimension z(1536),zg(1536),zc(1536),zgc(1536),iedge(100,2),
     1     xx(20,9),yy(20,9),distort(2,9),xr(24,3,2),yr(24,3,2),
     2     aa(20,9),bb(20,9),cc(20,9)
c---------------------------------------------------------------------




      
c (1) left and right edges for all channels in ordinates ja,jb 
c     Parameters:
      print *,'newgeom (2164)'
      im=1536
      jm=1024
      nm=9
c     1023 intervalles, 1022=2 fois 511 utilisables avec j=512 central
c     j=    1 / 512 / 1023
c     x=    0 / 511 / 1022
      jtriple=0   !1        !       j-1,j,j+1
      i1=5
      i2=im-4
      j1=1
      j2=jm
      call par1('     ja1',1,ja1)
      call par1('     ja3',1,ja3)
      ja(1)=ja1
      ja(3)=ja3
      ja(2)=(ja1+ja3)/2
c      ja(1)=1+150
c      ja(2)=1+500           ! 500 
c      ja(3)=1+850 
c             -----      
c      ja(2)=(ja(1)+ja(3))/2     ! jc= 512
      write(3,*)' newgeom: ja 1,2,3 ',(ja(nn),nn=1,3)
c     jc=(jm+1)/2   !  512,
      jc=ja(2)
      sig(1)=1.
      sig(2)=-1.
      call par1('   laddx',1,laddx)
      call par1('   laddy',1,laddy)
      call par1('     mgx',1,mgx)
      zgx=mgx   !  minimum threshold of gradient for the i of maximum intensity
      print *,' mgx,zxt',mgx,zgx
      call par1('     mgy',1,mgy)
      zgy=mgy   !  minimum threshold of gradient for the j of maximum intensity
      print *,' mgy,zgy',mgy,zgy    
c      call par1('  interp',1,interp)  !  parabolic interpolation with gradients
c-------------------
 1    format(10i6)
      write(3,*)'  Newgeom   meanflat:'
      write(3,1)(meanflat(i,512),i=1,1536,15)
      write(3,*)' edges for 3 j-values'
c----------------
c----------------                                 zmax,zgmax
c                                           d'après la coupe centrale j=jc
      zmax=0.
      do i=1,im
         zc(i)=meanflat(i,jc)       !  jc
         zmax=amax1(zc(i),zmax)
      enddo

       zgmax=0.
       do i=1,im-1
c         zmax=amax1(zmax,zc(i))
         zgc(i)=zc(i+1)-zc(i)
         piv=abs(zgc(i))
         zgmax=amax1(zgmax,piv)
      enddo
      write(3,*)'  zmax,zgmax ',zmax,zgmax
            do i=1,im
            zc(i)=100.*zc(i)/zmax
            zgc(i)=100.*zgc(i)/zgmax
      enddo
c------------------------------------------------- 3 coupes     

      do 30 nj=1,3               !  j de la coupe
      print *,'nj',nj
      jj=ja(nj)
      print *,'jj ',jj
      write(3,*)' '
      write(3,*)' ja(',nj,')= ',jj

      do i=i1,i2
      if(laddx.eq.0)then  ! ************
         z(i)=meanflat(i,jj)
      else
         z(i)=0
         jj1=jj-laddx
         jj2=jj+laddx
         do jjp=jj1,jj2
            z(i)=z(i)+meanflat(i,jjp) !   moyennes jadd
         enddo
         z(i)=z(i)/float(2*laddx+1)
      endif
      enddo

      do i=i1,i2-1
         zg(i)=z(i+1)-z(i)
c         piv=abs(zg(i))
c         zgmax=amax1(zgmax,piv)
      enddo
      zg(i2)=zg(i2-1)
      print *,' zmax,zgmax ',zmax,zgmax     !  valeurs pour coupe centrale

      do i=i1,i2
            z(i)=100.*z(i)/zmax
            zg(i)=100.*zg(i)/zgmax
      enddo
c------------------
c      print *,'(2212)'
c      if(nj.eq.2)then
c         do i=i1,i2
c            zc(i)=z(i)
c            zgc(i)=zg(i)
c         enddo
c      endif
c--------------
            write(3,*)'     '
            write(3,*)' newgeom: left and right edges of each n-channel'
 2       format(10f6.0)
      if(nj.eq.2)then
         write(3,*)' z(91 to 110) '
         write(3,2)(z(i),i=91,110)
         write(3,*)' zg(91 to 110) '
         write(3,2)(zg(i),i=91,110)
      endif

      n=1
      is=1
      i=i1
 10   continue
        i=i+1
        piv2=sig(is)*zg(i)
      if(piv2.lt.zgx)goto 10     !  *******************    max
        piv1=sig(is)*zg(i-1)
        piv3=sig(is)*zg(i+1)
        if(piv2.lt.piv1.or.piv2.lt.piv3)goto 10 ! **** piv1,piv3

        eps=0.5
c        if(interp.eq.1)
      call smax(zg,i,eps,apar,bpar,cpar)
 3      format(' edges: sig, l, n, iedge(n,is),zg(iedge-1/0/+1)',
     1     '  eps     XX      YY' )
 4    format(6x,f5.0,2i3,i6,5x,3f6.0,f6.2,2f8.2)
        if(is.eq.1)then
            l=nj
         else
            l=nj+3
         endif
      iedge(n,is)=i
      xx(l,n)=iedge(n,is)+eps-1.+0.5   !  xx instead of i, max parab
      yy(l,n)=ja(nj)-1.
      aa(l,n)=apar
      bb(l,n)=bpar
      cc(l,n)=cpar                      
      write(3,4)sig(is),l,n,iedge(n,is),zg(i-1),zg(i),zg(i+1),eps,
     1     xx(l,n),yy(l,n)
      write(3,*)' apar,bpar,cpar ',apar,bpar,cpar
      print *,' jj,is,n,mgx,zgt',jj,is,n,mgx,zgt
      
      if(is.eq.1)then
         is=2
         i=i+1
         goto 10
      endif

      if(is.eq.2)then
         n=n+1
         if(n.gt.nm)goto 30
         is=1
         i=i+1
         goto10
      endif
         
 30   enddo                     !  nj     coupe /         abcdef  canaux

      do n=1,nm
         xx(15,n)=xx(5,n)   ! E
         yy(15,n)=yy(5,n)
         xx(12,n)=xx(2,n)   ! B
         yy(12,n)=yy(2,n)
      enddo
c----------------------------      
c     distortion:  flèches entre abc   et    def
      valqm=0.
      dista1=0.
      dista2=0.
      do n=1,nm
         distort(1,n)=xx(2,n)-(xx(1,n)+xx(3,n))/2.
         distort(2,n)=xx(5,n)-(xx(4,n)+xx(6,n))/2.
         valqm=valqm+distort(1,n)**2+distort(2,n)**2
         dista1=dista1+distort(1,n)
         dista2=dista2+distort(2,n)
      enddo                     !    nm=9
      valqm=valqm/(2.*float(nm))
      valqm=sqrt(valqm)
      dista1=dista1/float(nm)
      dista2=dista2/float(nm)
 31   format(' distortion: ',f6.3,
     1      '  quadratic mean value in pixel-to-pixel distance ')
      write(3,31)valqm
 34   format('   mean values for 1 and 2 : ',2f7.2)
      write(3,34)dista1,dista2
 32   format('distortion: sig    n    distortion')
 33   format(10x,f5.0,i5,f8.2)
      write(3,32)
      is=1
      do n=1,nm
         write(3,33)sig(is),n,distort(1,n)
      enddo
      write(3,32)
      is=2
      do n=1,nm
         write(3,33)sig(is),n,distort(2,n)
      enddo
c-----------------------------------------------    points  k,l,m,n
      xdel=25.
      sig(1)=1.
      sig(2)=-1.
c     points k
      do 50 n=1,nm                                !   channels
         do 45 l=7,10              !  k,l,m,n
            if(l.eq.7)then         !  k   
               ii=xx(1,n)+1+xdel
               jj1=1
               jj2=yy(1,n)+1
               is=1
            endif
            if(l.eq.8)then          ! l
               ii=xx(3,n)+1+xdel
               jj1=yy(3,n)+1
               jj2=jm
               is=2
            endif
            if(l.eq.9)then          ! m
               ii=xx(4,n)+1-xdel
               jj1=1
               jj2=yy(4,n)+1
               is=1
            endif
            if(l.eq.10)then         ! n
               ii=xx(6,n)+1-xdel
               jj1=yy(6,n)+1
               jj2=jm
               is=2
            endif
c                                            z,zg            
      do jj=jj1,jj2
        if(laddy.eq.0)then    !  ***************
               z(jj)=meanflat(ii,jj)
        else
         z(jj)=0.
         ii1=ii-laddy
         ii2=ii+laddy
         do iip=ii1,ii2
            z(jj)=z(jj)+meanflat(iip,jj)   !   moyennes jadd
         enddo
         z(jj)=z(jj)/float(2*laddy+1)
      endif
      enddo

      do jj=jj1,jj2-1
         zg(jj)=(z(jj+1)-z(jj))*sig(is)   !   sign
c         piv=abs(zg(jj))
c         zgmax=amax1(zgmax,zg(jj))
      enddo
      zg(jj2)=zg(jj2-1)
c      print *,' l,n,zmax,zgmax ',l,n,zmax,zgmax
      do jj=jj1,jj2
            z(jj)=100.*z(jj)/zmax
            zg(jj)=100.*zg(jj)/zgmax
      enddo
c                                            zgmax           
      do 40 jj=jj1+1,jj2-1     
c            if(z(i).lt.zseuil)goto10  inutile 
        piv2=zg(jj)
      if(piv2.lt.zgy)goto 40                 !  zgt threshold
        piv1=zg(jj-1)
        piv3=zg(jj+1)
        if(piv2.lt.piv1.or.piv2.lt.piv3)goto40
        eps=0.5
c     if(interp.eq.1)
      call smax(zg,jj,eps,apar,bpar,cpar)
      xx(l,n)=ii-1
      yy(l,n)=jj-1+eps +0.5     ! yy instead of jj, max parab
      aa(l,n)=apar
      bb(l,n)=bpar
      cc(l,n)=cpar
      goto 45
 40   continue
 45   enddo                      !  l
 50   continue                   !  n
c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
      write(3,*)' interp ',interp
      do 60 n=1,nm
      x1=xx(2,n)  !1
      x2=xx(1,n)  !2
      x3=xx(7,n)  !7
      x4=xx(9,n)  !9
      y1=yy(2,n)
      y2=yy(1,n)
      y3=yy(7,n)
      y4=yy(9,n)
      call intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
      xx(11,n)=xres                                     !  A
      yy(11,n)=yres
 53   format(' A: interp,n, x1,x2,x2,x4,  y1,y2,y3,y4, xres,yres',
     1          /2i3,2x,4f7.1,2x,4f7.1,2x,2f7.1)
      write(3,53)interp,n,x1,x2,x3,x4,y1,y2,y3,y4,xres,yres     

      xx(12,n)=xx(2,n)                                  !  B
      yy(12,n)=yy(2,n)
  
      x1=xx(2,n)  !b    !2
      x2=xx(3,n)  !c    !3
      x3=xx(8,n)  !l
      x4=xx(10,n)  !n
      y1=yy(2,n)  
      y2=yy(3,n)
      y3=yy(8,n)
      y4=yy(10,n)
      call intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
      xx(13,n)=xres                                       !C   ?
      yy(13,n)=yres
 51   format(' C: interp,n, x1,x2,x2,x4,  y1,y2,y3,y4, xres,yres',
     1          /2i3,2x,4f7.1,2x,4f7.1,2x,2f7.1)
      write(3,51)interp,n,x1,x2,x3,x4,y1,y2,y3,y4,xres,yres
c--------               abcdef    klmn2     ABCDEF
c                       1    6    7  10   11    16
      x1=xx(5,n)  !d  e
      x2=xx(4,n)  !e  d   
      x3=xx(9,n)  !k  m
      x4=xx(7,n)  !m  k
      y1=yy(5,n)  
      y2=yy(4,n)
      y3=yy(9,n)
      y4=yy(7,n)
      call intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
      xx(14,n)=xres                                      !D
      yy(14,n)=yres

      xx(15,n)=xx(5,n)                                  !  E
      yy(15,n)=yy(5,n)
      
      x1=xx(5,n)  !e
      x2=xx(6,n)  !f      
      x3=xx(10,n)  !l   8
      x4=xx(8,n)  !n  10
      y1=yy(5,n)  
      y2=yy(6,n)
      y3=yy(10,n)
      y4=yy(8,n)
      call intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
      xx(16,n)=xres                                      !F   ?
      yy(16,n)=yres
 52   format(' F: interp,n, x1,x2,x2,x4,  y1,y2,y3,y4, xres,yres',
     1          /2i3,2x,4f7.1,2x,4f7.1,2x,2f7.1)
      write(3,52)interp,n,x1,x2,x3,x4,y1,y2,y3,y4,xres,yres
 60   continue
c-----------------------------------------
 65   format(9f8.2)
      write(3,*)' '
      write(3,*)' Points:           abcdef    klmn     ABCDEF'
      write(3,*)'        with nl=   1....6    7..10   11....16'  
      write(3,*)' Every both lines   xx(nl,nc) / yy(nl,nc)'
      write(3,*)'                    for all nc=channels'
      do 70 nl=1,16
      write(3,*)' nl=',nl
      write(3,65)(xx(nl,nc),nc=1,nm)
      write(3,65)(yy(nl,nc),nc=1,nm)
 70   continue
      
      write(3,*)' nexgeom:  first channel'
      do nn=1,6
         write(3,*)'  points: abcdef nn, xx,yy', nn,xx(nn,1),yy(nn,1)
      enddo
         write(3,*)' '
      do nn=7,10
         write(3,*)'  points: klmn  nn, xx,yy', nn,xx(nn,1),yy(nn,1)
      enddo
         write(3,*)' '
      do nn=11,16
         write(3,*)'  points: ABCDEF  nn, xx,yy', nn,xx(nn,1),yy(nn,1)
      enddo
         write(3,*)' ' 

      call plotgeo1(zc,zgc,grt,i1,i2,im,jm,nm,xx,yy,ja,aa,bb,cc)
      call plotgeo3(xx,yy,xdel)   !  ,xr,yr)

      call plotgeo4(xx,yy)

c     xx,yy converted to xr,yr
cCOEFF   n=           1
c n,kg,xr    1   1   61.63  505.14  948.65     (24,3,2)
c n,kg,yr    1   1  115.78   99.32   82.69
c n,kg,xr    1   2   59.06  503.83  948.59
c n,kg,yr    1   2  237.84  221.48  204.56

c   points: ABCDEF  nn, xx,yy          11   114.397163       61.0217705    
c   points: ABCDEF  nn, xx,yy          12   98.1159821       500.000000    
c   points: ABCDEF  nn, xx,yy          13   81.3592606       947.905151    
c   points: ABCDEF  nn, xx,yy          14   236.158035       58.3470688    
c   points: ABCDEF  nn, xx,yy          15   220.216690       500.000000    
c   points: ABCDEF  nn, xx,yy          16   202.995148       947.756653 
c                      (20,9)
         do n=1,nm
            do kg=1,2
               do l=1,3
                  lp=l+10+3*(kg-1)
                  xr(n,l,kg)=yy(lp,n)
                  yr(n,l,kg)=xx(lp,n)
               enddo
            enddo
         enddo
      write(3,*)' xr n=1 ',(xr(1,l,1),l=1,3),(xr(1,l,2),l=1,3)
      write(3,*)' yr n=1 ',(yr(1,l,1),l=1,3),(yr(1,l,2),l=1,3) 
      
      return
      end
c-------------------------------------
      subroutine intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
c     long edge         short edge
c     l=2,1             l=7,9
c      x1,y1  x2,y2     x3,y3  x4,y4
c      x=ay+b            y=cx+d
c     a is small        c may be zero
c     xres=(bc+d)/(1-ac)
c     yres=c*xres+d

c     y=cx+d   y3=cx3+d  y3-y4=c(x3-x4)    c=              d=y3-c*x3
c              y4=cx4+d
c     x=ay+b   x2=ay2+b  x2-x1=a(y2-y1)    a=              b=x1-a*y1
c              x1=ay1+b
c     x=a(cx+d)+b        x(1-ac)=ad+b      xres=(ad+b)/(1-ac)
c                                          yres=c*xres+d

      a=(x2-x1)/(y2-y1)               !     (x4-x3)/(y4-y3)
      b=x1-a*y1                       !          y3-a*x3
      c=(y3-y4)/(x3-x4)               !                   (y2-y1)/(x2-x1)
      d=y3-c*x3                       !         x2-c*y2
      ac=a*c                          !  2026-09-21 : ac était non déclaré (bug F77
                                      !  implicite = 0) ; le dénominateur réel est 1-a*c
      xres=(a*d+b)/(1.-ac)
      yres=c*xres+d
      write(3,*)' intersec: x1..x4 / y1..y4 / a,b,c,d, xres,yres'
 1    format(6f8.2)
      write(3,1)x1,x2,x3,x4
      write(3,1)y1,y2,y3,y4
      write(3,1)a,b,c,d,xres,yres   
      return
      end      
c-------------------------------------
c      subroutine intersec2(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)
c                         2  3  8 10  2  3  8 10
cc      a=(x2-x1)/(y2-y1)
cc      b=(y4-y3)/(x4-x3)
cc      xres=x2+a*(x2+b*x3)/(a*b-1.)
cc      yres=y3+b*(y3+b*y2)/(a*b-1.)
cc      return

c      ayx=(y2-y1)/(x2-x1)
c        byx=(y4-y3)/(x4-x3)
c        axy=1./ayx
c        bxy=1./byx

c      xres=(x3*byx-x1*ayx+y1-y3)/(byx-ayx)
c      yres=(y3*bxy-y1*axy+x1-x3)/(bxy-axy)
c      write(3,*)'intersec: axy,bxy  xres,yres ',axy,bxy,xres,yres
c      return
c      end
c========================================================
      subroutine  plotgeo1(zc,zgc,zgt,i1,i2,im,jm,nm,xx,yy,ja,aa,bb,cc)
      dimension zc(1536),zgc(1536),x(2),y(2),xdes(1536),ydes(1536),
     1          xgdes(1536),xlim(2),ylim(2),xpar(3),ypar(3)
c               shift 0.5 pixel
      dimension xx(20,9),yy(20,9),ides(7),ja(3),
     1          aa(20,9),bb(20,9),cc(20,9)
      data ides /11,12,13,16,15,14,11/

c      do n=1,nm
c         do ii=1,7
c            ip=ides(ii)
c            xdes(ii)=xx(ip,n)
c            ydes(ii)=yy(ip,n)
c         enddo
c         call pgsls(1)
c         call pgline(7,xdes,ydes)
c      enddo
c--------------------------------------   letters
c      call pgslw(3)
c      call pgsch(1.)
c      x=xx(1,1)-30     ! +10
c      y=yy(1,1)+10
c      call pgtext(x,y,'a')
c      x=xx(2,1)-30
c      y=yy(2,1)+10


      
      print *,i1,i2,im,jm
      do i=1,im
         xdes(i)=i-1
         xgdes(i)=xdes(i)+0.5              !  shift for gradient geo1.ps
      enddo
      im1=im-1
c-----------------------------------------------  plot geo1.ps
      call pgbegin(0,'geo1.ps/ps',1,1)
      call pgvport(0.1,0.6,0.7,0.9)  !                       (0.1,0.9,0.5,0.9)
         call pgslw(3)
         call pgsch(1.)
         call pgsls(1)      
                 x1=0.  !950.  !0.
                 x2=1536.-1.    !1010.      !1536.-1.
                 y1=0.
                 y2=jm-1
      call pgwindow(x1,x2,0.,100.)
      call pgbox('bcts',200.,2,'bcnts',50.,5) 
      call pglabel(' ','Intensity',
     1    'Cross-section along the center part of Y field of view')
c        piv=float(si)+5.
c     call pgtext(120.,piv,'sj')
      call pgline(im,xdes,zc)                    !   intensity curve
c------------      
      call pgvport(0.1,0.6,0.5,0.7)  !                      (0.1,0.9,.1,.5)
      call pglabel('X (unit=arcsec/2) ','Intensity gradient',' ')
      call pgwindow(x1,x2,-100.,100.)
      call pgbox('abcnts',200.,2,'bcnts',50.,5)

c     thresholds for gradient
      call par1('     mgx',1,mgx)
      zgx=mgx
      call pgsls(2)
      x(1)=0.5
      x(2)=im-1+0.5
      y(1)=zgx                  !  +
      y(2)=y(1)
      call pgline(2,x,y)
      y(1)=-y(1)               !  - 
      y(2)=y(1)
      call pgline(2,x,y)
c     thresholds for gradient
      print *,'  mgx ',mgx
      ypiv=zgx+15.
c              sgi2=sgi+10
c        sgi3=-sgi-25
      call pgsch(1.)
      call pgtext(120.,ypiv,'mgx')
      ypiv=-zgx-20.
      call pgtext(30.,ypiv,'-mgx')
      call pgsch(1.)

      call pgsls(1)
      x(2)=2
      
      call pgline(im,xgdes,zgc)    !  gradient curve
c      call pgvport(0.1,0.6,0.1,0.4)
c-------------------------------------------9 channels
      call pgvport(0.1,0.6,0.1,0.4)  
      call pgwindow(x1,x2,y1,y2)
      call pgbox('abcnts',200.,2,'bcnts',200.,2)

c      do n=1,nm
c         do id=1,7
c            ip=ides(id)
c            xdes(id)=xx(ip,n)
c            ydes(id)=yy(ip,n)
c         enddo
c         call pgsls(1)
c         call pgline(7,xdes,ydes)
c      enddo
c-----------------------------------------------
      write(3,*)'  ABCDEF X for nm channels'
 10   format(i4,2x,3f9.2,4x,3f9.2)
      do n=1,nm
         write(3,10)n,(xx(nd,n),nd=11,16)
      enddo
      write(3,*)'  ABCDEF Y for nm channels'
      do n=1,nm
         write(3,10)n,(yy(nd,n),nd=11,16)
      enddo
c--------------------------------------------file ACDF2.lis
      call system('rm ACDF2.lis')
      open(unit=20,file='ACDF2.lis',status='new')
 91   format(8f8.2)
      do n=1,nm
      write(20,91)xx(11,n),xx(13,n),xx(14,n),xx(16,n),
     1            yy(11,n),yy(13,n),yy(14,n),yy(16,n)
      enddo
      close(20)
c--------------------------------------------------------
c-------------------------------------------------ABCDEF
c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
c     lines
      do n=1,nm
      do nd=1,3
         np=nd+10
        xdes(nd)=xx(np,n)
        ydes(nd)=yy(np,n)
      enddo
      do nd=4,6
         np=20-nd
         xdes(nd)=xx(np,n)
         ydes(nd)=yy(np,n)
      enddo
      xdes(7)=xx(11,n)
      ydes(7)=yy(11,n)
      do nd=1,7
         write(3,*)' nd, xdes, ydes ',nd,xdes(nd),ydes(nd)
      enddo
      call pgsls(1)
      call pgline(7,xdes,ydes)        !            channels edges
      enddo

      xdes(1)=0.
      xdes(2)=1536.-1.
      do nd=1,3
         ydes(1)=ja(nd)-1.
         ydes(2)=ydes(1)
         call pgsls(2)
         call pgline(2,xdes,ydes)
      enddo
      call pgend
      call par1('   igeo1',1,igeo1)
      if(igeo1.eq.1)call system('okular geo1.ps &')
c====================================  plot geo2.ps
      call pgbegin(0,'geo2.ps/ps',1,1)
      call pgvport(0.1,0.6,0.6,0.9)  !   0.1,0.6,          (0.1,0.9,0.5,0.9)
         call pgslw(3)
         call pgsch(1.)
         call pgsls(1)
         xlong=6.
         xmax=xx(2,1)
         kmax=xmax
                 x1=kmax-xlong+0.5  !0.
                 x2=kmax+xlong+0.5         !1536.-1.
      write(3,*)' xmax,kmax,x1,x2  ',xmax,kmax,x1,x2
      call pgwindow(x1,x2,0.,100.)
      call pgbox('bcts',5.,5,'bcnts',50.,5)   !         200.,2,'bcnts',50.,5) 
      call pglabel(' ','Intensity',
     1    'Cross-section along the center part of Y field of view')
c        piv=float(si)+5.
c     call pgtext(120.,piv,'sj')
            do i=1,120    !   120 are sufficient
         xdes(i)=i-1
         xgdes(i)=xdes(i)+0.5              !  shift for gradient geo1b.ps
      enddo
      write(3,*)' zc(i),i=96,99 ',(zc(i),i=96,99)
      call pgline(120,xdes,zc)   !   intensity curve
      xmax=xx(2,1)              !  X of max  first channel  b
      xdes(1)=xmax
      xdes(2)=xmax
      ydes(1)=0.
      ydes(2)=100.
      call pgline(2,xdes,ydes)   !  vertical line xmax

      call pgsls(2)
      lim1=xdes(2)-1
      xlim(1)=lim1
      xlim(2)=lim1
      call pgline(2,xlim,ydes)
      xlim(1)=xlim(1)+3.
      xlim(2)=xlim(2)+3.
      call pgline(2,xlim,ydes)   ! vertical lines
      call pgsls(1)
c------------      
      call pgvport(0.1,0.6,0.3,0.6)  !                      (0.1,0.9,.1,.5)
      call pglabel('X (unit=arcsec/2) ','Intensity gradient',' ')
      call pgwindow(x1,x2,-100.,100.)
      call pgbox('abcnts',5.,5,'bcnts',50.,5)   ! 200.,2,'bcnts',50.,5)
        call pgsch(1.)
c      call pgtext(120.,grt,'grt')   !  non c

        xmax=xx(2,1)              !  X of max  first channel  b
      xdes(1)=xmax
      xdes(2)=xmax
      ydes(1)=-100.
      ydes(2)=100.
      call pgline(2,xdes,ydes)   !  vertical line xmax   gradient
   
      call pgsls(2)
      xlim(1)=lim1
      xlim(2)=lim1
      call pgline(2,xlim,ydes)   !  vertical lines
      xlim(1)=xlim(1)+3.
      xlim(2)=xlim(2)+3.
      call pgline(2,xlim,ydes)

      call par1('     mgx',1,mgx)
        zgx=mgx
      x(1)=0
      x(2)=120.   !im-1
      y(1)=zgx                  !  +
      y(2)=y(1)
      call pgsls(2)
      call pgline(2,x,y)
      y(1)=-y(1)               !  - 
      y(2)=y(1)
      call pgline(2,x,y)
c                                  thresholds for gradient
      call pgsch(1.)
      xpiv=xmax-4.5
      ypiv=zgx+8.
      call pgtext(xpiv,ypiv,'mgx')
      xpiv=xpiv-0.3
      ypiv=-zgx-15.
      call pgtext(xpiv,ypiv,'-mgx')  
      call pgsch(1.)
      call pgslw(3)
      call pgsls(1)

c      xmax=xx(2,1)
c                   !  interpol parab
c      xdes(2)=kmax+0.5
c      xdes(1)=kmax-0.5
c      xdes(3)=kmax+1.5
c      ydes(2)=zc(kmax+1)-zc(kmax)
c      ydes(1)=zc(kmax)-zc(kmax-1)
c      ydes(3)=zc(kmax+2)-zc(kmax+1)
c      write(3,*)' xx(2,1), kmax, zc(kmax) ',xx(2,1), kmax, zc(kmax)

      call pgline(150,xgdes,zgc)
c     do k=1,3
c         call pgpt(3,xdes,ydes,ICHAR('X'))
c     enddo
      piv=xmax+0.1  !-0.25
c     call pgptxt(piv,-80.,90.0,0.,'B pt')
      call pgtext(piv,-90.,'B')

       xpar(2)=kmax+0.5-0.22      !   3X parabole first channel
       kpiv=kmax+1
       ypar(2)=zgc(kpiv)-9
       xpar(1)=xpar(2)-1
       xpar(3)=xpar(2)+1
       ypar(1)=zgc(kpiv-1)-9
       ypar(3)=zgc(kpiv+1)-9
       call pgsch(1.5)
       do ipar=1,3
          call pgtext(xpar(ipar),ypar(ipar),'X')
       enddo
       apar=aa(2,1)
       bpar=bb(2,1)
       cpar=cc(2,1)
       do ipar=1,21
          xdes(ipar)=kmax+0.5+0.1*(ipar-11)
            xpiv=0.1*(ipar-11)
          ydes(ipar)=cpar+xpiv*(bpar+xpiv*apar)
       enddo
       write(3,*)' aa,bb,cc (2,1)  xpar (1,2,3)'
       write(3,*)aa(2,1),bb(2,1),cc(2,1)
       write(3,*)xpar(1),xpar(2),xpar(3)
       call pgsls(1)
       call pgslw(5)
       call pgline(21,xdes,ydes)
c------------------------------------------
       call pgend
       call par1('   igeo2',1,igeo2) 
      if(igeo2.eq.1)call system('okular geo2.ps &')
c------------------------------------------    
      return
      end
c======================================
      subroutine plotgeo3(xx,yy,xdel)
c                               +/-decalages en X pour klmn      
c     dimension  xr(24,3,2),yr(24,3,2),xdes(7),ydes(7)
c      dimension  xr(24,3,2),yr(24,3,2)
      dimension xx(20,9),yy(20,9),xdes(7),ydes(7)
c      xdel=25   !shift for k,l,m,n
      
      n=1

      call pgbegin(0,'geo3.ps/ps',1,1)
      call pgvport(0.1,0.3,0.1,0.9)
         call pgslw(3)
         call pgsch(1.)
         call pgsls(1)      
                 x1=0.
                 x2=350.
      call pgwindow(x1,x2,0.,1023.)
      call pgbox('bcnts',100.,2,'bcnts',200.,2) 
      call pglabel('X','Y','First channel')
c      xdes(1)=xr(n,1,1)
c      ydes(1)=yr(n,1,1)
c      xdes(2)=xr(n,2,1)
c      ydes(2)=yr(n,2,1)
c      xdes(3)=xr(n,3,1)
c      ydes(3)=yr(n,3,1)
c      xdes(4)=xr(n,3,2)
c      ydes(4)=yr(n,3,2)
c      xdes(5)=xr(n,2,2)
c      ydes(5)=yr(n,2,2)
c      xdes(6)=xr(n,1,2)
c      ydes(6)=yr(n,1,2)
c      xdes(7)=xdes(1)
c      ydes(7)=ydes(1)
c      write(3,*)'plot geo2.ps   '
c 203  format(2i4,6f8.1)
c      write(3,*)'Left,Right  n,kg,x,x,y,y ',n,kg,x(1),x(2),y(1),y(2)
c      write(3,*)'Left,Right   n,kg, xdes / ydes'
c      write(3,203) n,kg,(xdes(nn),nn=1,6)
c      write(3,203) n,kg,(ydes(nn),nn=1,6)
c      call pgsls(1)    !(4)
c      call pgline(7,xdes,ydes)  ! contour channel
      call pgsls(4)
      call pgslw(3)
      do nn=1,3                     !  Y=Y1, Y2, Y3
         xdes(1)=x1
         xdes(2)=x2
         ydes(1)=yy(nn,1)   !yr(1,2,1)
         ydes(2)=ydes(1)
      call pgline(2,xdes,ydes)
      enddo
      call pgsls(1)
      call pgslw(3)
c-----------------------------------      
      do nn=1,6
c      call pgpoint(1,xdes(nn),ydes(nn),8)
      call pgpt(6,xx(nn,1),yy(nn,1),8)
      enddo
      write(3,*)' geo2:'
      write(3,*)'xx / yy  for first channel'
      write(3,*)(xx(nn,1),nn=1,6)
      write(3,*)(yy(nn,1),nn=1,6)
c-----------------------------k,l,m,n
      call pgsls(4)
      call pgslw(3)
      xdes(1)=xx(1,1)+xdel
      xdes(2)=xdes(1)
      ydes(1)=0.
      ydes(2)=yy(1,1)
      call pgline(2,xdes,ydes)
      
      xdes(1)=xx(3,1)+xdel
      xdes(2)=xdes(1)
      ydes(1)=yy(3,1)
      ydes(2)=1023.
      call pgline(2,xdes,ydes)

      xdes(1)=xx(4,1)-xdel
      xdes(2)=xdes(1)
      ydes(1)=0.
      ydes(2)=yy(1,1)
      call pgline(2,xdes,ydes)

      xdes(1)=xx(6,1)-xdel
      xdes(2)=xdes(1)
      ydes(1)=yy(6,1)
      ydes(2)=1023.
      call pgline(2,xdes,ydes)
c-------------------------------------
      do klmn=7,16
         xdes(1)=xx(klmn,1)
         ydes(1)=yy(klmn,1)
         call pgpt(1,xdes(1),ydes(1),8)
      enddo
c-------------------------------------------------ABCDEF
c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
c     lines
      do nd=1,3
         np=nd+10
        xdes(nd)=xx(np,1)
        ydes(nd)=yy(np,1)
      enddo
      do nd=4,6
         np=20-nd
         xdes(nd)=xx(np,1)
         ydes(nd)=yy(np,1)
      enddo
      xdes(7)=xx(11,1)
      ydes(7)=yy(11,1)
      do nd=1,7
         write(3,*)' nd, xdes, ydes ',nd,xdes(nd),ydes(nd)
      enddo
      call pgsls(1)
      call pgline(7,xdes,ydes)
c--------------------------------------   letters
      call pgslw(3)
      call pgsch(1.)
      x=xx(1,1)-30     ! +10
      y=yy(1,1)+10
      call pgtext(x,y,'a')
      x=xx(2,1)-30
      y=yy(2,1)+10
      call pgtext(x,y,'b')
      x=xx(3,1)-25               !+10
      y=yy(3,1)-30
      call pgtext(x,y,'c')

      x=xx(4,1)+10          !-30
      y=yy(4,1)+10
      call pgtext(x,y,'d')
      x=xx(5,1)+10
      y=yy(5,1)+10
      call pgtext(x,y,'e')
      x=xx(6,1)+10
      y=yy(6,1)-30              !-20
      call pgtext(x,y,'f')
c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
      x=xx(7,1)+10          
      y=yy(7,1)-30
      call pgtext(x,y,'k')
      x=xx(8,1)+10
      y=yy(8,1)+10
      call pgtext(x,y,'l')
      x=xx(9,1)-35            !-25
      y=yy(9,1)-30             
      call pgtext(x,y,'m')
      x=xx(10,1)-30
      y=yy(10,1)+10             
      call pgtext(x,y,'n')
      
      x=xx(11,1)-30          
      y=yy(11,1)-30
      call pgtext(x,y,'A')
      x=xx(12,1)-30
      y=yy(12,1)-30
      call pgtext(x,y,'B')
      x=xx(13,1)-30            !-25
      y=yy(13,1)+10             
      call pgtext(x,y,'C')
      x=xx(14,1)+10
      y=yy(14,1)-30             
      call pgtext(x,y,'D')
      x=xx(15,1)+10          
      y=yy(15,1)-30
      call pgtext(x,y,'E')
      x=xx(16,1)+10
      y=yy(16,1)+10
      call pgtext(x,y,'F')
      
c---------------------------------------
      call pgend
      call par1('   igeo3',1,igeo3) 
      if(igeo3.eq.1)call system('okular geo3.ps &')
      return
      end
c======================================
  
c     call pgvport(0.1,0.9,.1,.5)
c      call pglabel('X (unit=arcsec/2) ','Intensity gradient',' ')
c      call pgwindow(x1,x2,-100.,100.)
c      call pgbox('abcnts',200.,2,'bcnts',50.,5)

c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16


c--------------------------------------   letters
c      call pgslw(3)
c      call pgsch(1.)
c      x=xx(1,1)-30     ! +10
c      y=yy(1,1)+10
c      call pgtext(x,y,'a')
c      x=xx(2,1)-30
c      y=yy(2,1)+10

c======================================
      subroutine plotgeo4(xx,yy)    
      dimension xx(20,9),yy(20,9),xdes(9),ydes(9)
      nm=9
      nc=5
      xnm1=nm+1.
      write(3,*)' plotgeo4: '
c      call par1('  interp',1,interp)     
c      if(interp.eq.1)call pgbegin(0,'geo4.ps/ps',1,1)
c      if(interp.eq.0)call pgbegin(0,'geo3b.ps/ps',1,1)
      call pgbegin(0,'geo4.ps/ps',1,1)
         call pgslw(3)   !(2)
         call pgsch(1.)
         call pgsls(1)      
      call pgvport(0.25,0.75,0.1,0.9)             !       (0.1,0.6,0.1,0.9)
      call pgvport(0.25,0.45,0.1,0.9)             !      (0.1,0.3,0.1,0.9)
      call pglabel(' ',' ','X')
      call pgvport(0.55,0.75,0.1,0.9)              !     ,0.6,0.1,0.9)
      call pglabel(' ',' ','Y')
c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
c     X      
c     AC
      call pgvport(0.25,0.45,0.7,0.9)  
      do n=1,nm
         ydes(n)=abs(xx(11,n)-xx(13,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: AC ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      call pgsch(1.5)                          !1.5
      xt=2.
      yt=ywin2-2.
      call pgtext(xt,yt,'AC')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)                           ! 1.
      call pgline(9,xdes,ydes)
      
c     DF
c     call pgvport(0.1,0.3,0.5,0.7)
      call pgvport(0.25,0.45,0.5,0.7)
            do n=1,nm
         ydes(n)=abs(xx(14,n)-xx(16,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: DF ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      xt=2.
      yt=ywin2-2.
      call pgsch(1.5)
      call pgtext(xt,yt,'DF')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)
      
c     AD      
c     call pgvport(0.1,0.3,0.3,0.5)
      call pgvport(0.25,0.45,0.3,0.5)      
      do n=1,nm
         ydes(n)=abs(xx(11,n)-xx(14,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: AC ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      xt=2.
      yt=ywin2-2.
      call pgsch(1.5)
      call pgtext(xt,yt,'AD')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)
      
c     CF
c     call pgvport(0.1,0.3,0.1,0.3)
      call pgvport(0.25,0.45,0.1,0.3)      
      do n=1,nm
         ydes(n)=abs(xx(13,n)-xx(16,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: CF ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      xt=2.
      yt=ywin2-2.
      call pgsch(1.5)
      call pgtext(xt,yt,'CF')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)
      
c     Y
c     AC
      call pgvport(0.55,0.75,0.7,0.9)  
      do n=1,nm
         ydes(n)=abs(yy(11,n)-yy(13,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: AC ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      call pgsch(1.5)
      xt=2.
      yt=ywin2-2.
      call pgtext(xt,yt,'AC')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)


c     DF
c     call pgvport(0.4,0.6,0.5,0.7)
      call pgvport(0.55,0.75,0.5,0.7)        
            do n=1,nm
         ydes(n)=abs(yy(14,n)-yy(16,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: DF ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      call pgsch(1.5)
      xt=2.
      yt=ywin2-2.
      call pgtext(xt,yt,'DF')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)

      
c     AD      
c     call pgvport(0.4,0.6,0.3,0.5)
      call pgvport(0.55,0.75,0.3,0.5)        
      do n=1,nm
         ydes(n)=abs(yy(11,n)-yy(14,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: AC ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      call pgsch(1.5)
      xt=2.
      yt=ywin2-2.
      call pgtext(xt,yt,'AD')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)
      
c     CF
c     call pgvport(0.4,0.6,0.1,0.3)
      call pgvport(0.55,0.75,0.1,0.3)        
      do n=1,nm
         ydes(n)=abs(yy(13,n)-yy(16,n))
         xdes(n)=n
      enddo
      write(3,*)' plotgeo3: CF ',(ydes(n),n=1,nm)
      ywin1=ydes(nc)-5.
      ywin2=ywin1+10.
      call pgwindow(0.,xnm1,ywin1,ywin2)
      call pgbox('bcs',10.,1,'bcnts',5.,5)
      call pgsch(1.5)
      xt=2.
      yt=ywin2-2.
      call pgtext(xt,yt,'CF')
      do n=1,nm
      call pgpoint(1,xdes(n),ydes(n),5)
      enddo
      call pgsch(1.)
      call pgline(9,xdes,ydes)
      
      call pgend
      call par1('   igeo4',1,igeo4) 
      if(igeo4.eq.1)call system('okular geo4.ps &')
      return
      end
c=============================================
c     call pgvport(0.1,0.9,.1,.5)
c      call pglabel('X (unit=arcsec/2) ','Intensity gradient',' ')
c      call pgwindow(x1,x2,-100.,100.)
c      call pgbox('abcnts',200.,2,'bcnts',50.,5)

c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16
c*********************************************************
      subroutine channels(xr,yr,ima,imima,jmima,cymx,iim,jjm,nm)
c                         in    in              out              1 = flat1.ps
c                                                                2 = flat2.ps
      dimension cymx(2000,200,24)
      integer*4 ima(1536,1536)
      dimension xr(24,3,2),yr(24,3,2)
      dimension ich(6),jch(6),xch(6),ych(6),xima(6),yima(6) !  ichannel,jchannel
      double precision coef(6,2)

      print *,' '
      print *,'                             CHANNELS'
      call par1('  milsec',1,milsec)
      call par1('      li',1,li)
      call par1('      lj',1,lj)
          iim=float(li)/float(milsec)+1.5     ! dim channels
          jjm=float(lj)/float(milsec)+1.5
 1    format(' imima,jmima,  li,lj,  iim,jjm ',6i7)
      print 1,imima,jmima,li,lj,iim,jjm
      
      print *,' ima(i,j)  200,200',imima,jmima
 2    format(i4,2x,10i4)
      do j=1,jmima,200
         print 2,j,(ima(i,j),i=1,imima,200)
      enddo

c     CCD        imb    1536     jmb   1024
c     permut     imc    1024     jmc   1536
c     channels   imd     885     jmd    123     nm   9
      do n=1,nm
      print *,' '
      print *,'COEFF   n=',n
      write(3,*)'COEFF   n=',n
 21   format(' n,kg,xr ',2i4,3f8.2)
 22   format(' n,kg,yr ',2i4,3f8.2)
        do kg=1,2
           write(3,21)n,kg,(xr(n,ihb,kg),ihb=1,3)
           write(3,21)n,kg,(yr(n,ihb,kg),ihb=1,3)
        enddo
      enddo
      
      do n=1,9                        
         kdistor=1
      write(3,*)' 1619 jjm ',jjm
      call COEFF(iim,jjm,xr,yr,n,coef,kdistor) 
c         print *,'yr(9,3,2)',yr(9,3,2)         
       do jj=1,jjm        
          do ii=1,iim     
             xx=ii-1.
             yy=jj-1.

       call PIX(xx,yy,coef,i,j,di,dj)        !  msdp2 2340
c     ---  in         out       
 3     format(' coord channel, coord file y  ',4f7.1)

       if(ii.eq.iim.and.jj.eq.jjm)then
          xxima=i+di-1
          yyima=j+dj-1
          print *,' Extremes points channel ',n
          print 3,xx,yy,xxima,yyima

          write(3,*)' Extremes points channel ',n
          write(3,3)xx,yy,xxima,yyima
          write(3,*)' '
       endif

       cymx(ii,jj,n)=(ima(i,j)*(1.-di)+ima(i+1,j)*di)*(1.-dj)
c            iitab,jjtab
     1      +(ima(i,j+1)*(1.-di)+ima(i+1,j+1)*di)*dj

      enddo    !  jj
      enddo    !  ii

      write(3,*)' cymx channel   n, iim,jjm  ',n, iim, jjm
 23   format(i4,2x,15f5.0)
      do ii=1,iim,200
         write(3,23)ii,(cymx(ii,jj,n),jj=1,jjm,10)
      enddo
      enddo                     ! n
      call map3(cymx,iim,jjm,nm)                            !  flat2.ps
      write(3,*)' sortie channels'
      write(3,*)' end channels  iim,jjm,nm ', iim,jjm,nm
      return
      end
c********************************************************************
      subroutine COEFF(im,jm,xr,yr,n,coef,kdistor)
      dimension xr(24,3,2),yr(24,3,2)
      double precision coef(6,2),xl,yl,x1,x2,x3,x4,x5,x6,
     1     y1,y2,y3,y4,y5,y6
      kdistor=0
      
      print *,' '
      print *,'COEFF   n=',n
 1    format(' n,kg,xr ',2i4,3f8.2)
 2    format(' n,kg,yr ',2i4,3f8.2)
        do kg=1,2
           print 1,n,kg,(xr(n,ihb,kg),ihb=1,3)
           print 2,n,kg,(yr(n,ihb,kg),ihb=1,3)
        enddo
      
      xl=im-1
      yl=jm-1
c      print *,'xl,yl',xl,yl
c
      x1=xr(n,1,1)
      x2=xr(n,3,1)
      x3=xr(n,1,2)
      x4=xr(n,3,2)
      x5=xr(n,2,1)
      x6=xr(n,2,2)

      y1=yr(n,1,1)
      y2=yr(n,3,1)
      y3=yr(n,1,2)
      y4=yr(n,3,2)
      y5=yr(n,2,1)
      y6=yr(n,2,2)
c
      coef(1,1)=x1
      coef(3,1)=(x3-x1)/yl

      if(kdistor.eq.0)then
      coef(2,1)=(x2-x1)/xl
      coef(4,1)=(x1+x4-x2-x3)/(xl*yl)
      coef(5,1)=0.
      coef(6,1)=0.
      else
      coef(2,1)=(-3.*x1-x2+4.*x5)/xl
      coef(4,1)=(3.*x1+x2-3.*x3-x4-4.*x5+4.*x6)/(xl*yl)
      coef(5,1)=2.*(x1+x2-2.*x5)/(xl*xl)
      coef(6,1)=2.*(-x1-x2+x3+x4+2.*x5-2.*x6)/(xl*xl*yl)
      endif
c
      coef(1,2)=y1
      coef(3,2)=(y3-y1)/yl

      if(kdistor.eq.0)then
      coef(2,2)=(y2-y1)/xl
      coef(4,2)=(y1+y4-y2-y3)/(xl*yl)
      coef(5,2)=0.
      coef(6,2)=0.
      else
      coef(2,2)=(-3.*y1-y2+4.*y5)/xl
      coef(4,2)=(3.*y1+y2-3.*y3-y4-4.*y5+4.*y6)/(xl*yl)
      coef(5,2)=2.*(y1+y2-2.*y5)/(xl*xl)
      coef(6,2)=2.*(-y1-y2+y3+y4+2.*y5-2.*y6)/(xl*xl*yl)
      endif
 3    format(6f12.6)
      print *,'n,coef ',n
      print 3,(coef(k,1),k=1,6)
      print 3,(coef(k,2),k=1,6)
      return
      end
c--------------
      subroutine pix(x,y,coef,i,j,di,dj)
c                ---
      double precision coef(6,2),xx,yy
        xx=x
        yy=y
      xi=coef(1,1)+xx*coef(2,1)+yy*coef(3,1)+xx*yy*coef(4,1)+
     1   (xx*xx)*(coef(5,1)+yy*coef(6,1))
      yj=coef(1,2)+xx*coef(2,2)+yy*coef(3,2)+xx*yy*coef(4,2)    !+

      i=xi
      j=yj
      di=xi-i
      dj=yj-j
      return
      end      
c********************************

c==================================================
      subroutine calib(cymx,imd,jmd,nm,cliss,jtr,cal,xr,yr)
c                                                calibration
      dimension cymx(2000,200,24)
      dimension cliss(2000,200,24),des(200,2000)
c                                 plot cal2      
      integer*4 iliss,kj(200),nplis(2000,200,24),jplis(2000,200,24)
      character*9 titre1,titre2
      character*100 titre
      dimension profmm(2000),dyln(200,24),yln(2000,24),    !    yln???
     1     prof(2000,100,24),profm(2000,24),center(2000,24),    !  2000 ?
     2     i1tr(24),i2tr(24),              calfac(2000,200,24),
     3     x(2000),y(2000),z(2000),pte(24),cal(2000,200,24),
     4     profij(2000,200,24),coef(2000,24)
      dimension xr(24,3,2),yr(24,3,2)
      dimension sobs(2000,200,24),sobsD1(2000,200,24),
     1     sobsD2(2000,200,24), obs(200,2000),tr(6)
      dimension sobsplot(2000,200,24),sobstot(2000,200,24,10)

        tr(1)=0. 
        tr(2)=1.
        tr(3)=0.
        tr(4)=0.
        tr(5)=0.
        tr(6)=1.

      call par1('   jliss',1,jliss)
      
      write(3,*)' entree calib'
      call par1('      nr',1,nr)
      ncurv=nr
      ncurv1=nr
      ncurv2=nr+1
      
c plot   calib.ps
      call pgbegin(0,'calib.ps/ps',2,3)  !3)
c calcul cliss   lissage/i      
      call par1('margline',1,margline)
      call par1('   iliss',1,iliss)
      call par1('   jliss',1,jliss)
      call par1('  jtrans',1,jtr)
      i1=1+margline          !  1+20
      i2=imd-margline
      j1=1+margline          !  1+20
      j2=jmd-margline
      il1=i1+iliss       !****
      il2=i2-iliss        !****
      jl1=j1+jliss
      jl2=j2-jliss
      print *,'j2 ',j2
      ic=(i1+i2)/2
      jc=(j1+j2)/2
      ic1=ic-1
      nc=(1+nm)/2
      den=(2*iliss+1)*(2*jliss+1)
      print *,' calib'
      print *,' imd,jmd,nm  i1,i2,iliss,jliss,il1,il2,ic j1,j2 den '
      write(3,*)' imd,jmd,nm,i1,i2,iliss,jliss,il1,il2,ic,j1,j2, den'
      write(3,1)imd,jmd,nm,i1,i2,iliss,jliss,il1,il2,ic, j1,j2, den
 1    format(12i5,f6.1)
      print 1,imd,jmd,nm,i1,i2,iliss,jliss,il1,il2,ic,j1,j2,den
c             cliss
      do n=1,nm
c      write(3,*)' n, cymx(1,1,n)',n, cymx(1,1,n)
        do j=1,jmd
        do i=1,imd
           cliss(i,j,n)=cymx(i,j,n)
        enddo
        enddo
      
      do j=jl1,jl2
         do i=il1,il2
            piv=0.
            do jp=j-jliss,j+jliss
            do ip=i-iliss,i+iliss
               piv=piv+cymx(ip,jp,n)             !  moy
            enddo
            enddo
            cliss(i,j,n)=piv/den
c           ------------- 
         enddo
      enddo
      write(3,*)' before plot_line'
      do i=1,imd,200
      write(3,*)'1897 n,i, cymx(i,j,n) ',n,i,(cymx(i,j,n),j=1,jmd,20)   ! B
      write(3,*)' 1897 n,i, cliss(i,j,n) ',n,i,(cliss(i,j,n),j=1,jmd,20) !  B
      enddo
      enddo                     !  n

c      call trans(cliss,imd,jmd,nm,jtrcalc,devcalc)
c     -----   translation of wavelengths between channels
c-------------------------------------------------------
      call par1('  ntrans',1,ntrans)
      trjm1=0.
      if(ntrans.ne.0)then
         call transpec(ntrans,xr,yr,imd,jmd,nm,trjm1)
c         jtr=trjm1+0.5
      endif
      write(3,*)' transpec: ntrans,trjm1 ',ntrans,trjm1
c----------------------------------------------------plot line centers
      call plot_line(trjm1,cymx,imd,jmd,i1,i2,iliss,jliss,
     1     il1,il2,jl1,jl2,nm,cliss,trjm,yln,center,pte,jtr)
c                                   out
c      call par1('      jt',1,jt)
c     if(jt.ne.0)trjm=jt
c      call par1('   ncurv',1,ncurv)
      write(3,*)' after plot_line: center(i,ncurv) ',
     1     (center(i,ncurv),i=1,imd,100)
c----------------------------------------mean profile
      xc=float(1+jmd)/2.
      jt1=xc-float(jtr)/2.+0.5
      jt2=jt1+jtr
      print *,' before profmean: xc,jt1,jt2,jtr ',xc,jt1,jt2,jtr
      write(3,*)' before profmean: xc,jt1,jt2,jtr ',xc,jt1,jt2,jtr
c------------------------------------      
      do i=1,imd,200
      write(3,*)'  1927 5,i, cymx(i,j,n) ',n,i,(cymx(i,j,5),j=1,jmd,20) ! B
      write(3,*)' 1927 5,i, cliss(i,j,n) ',n,i,(cliss(i,j,5),j=1,jmd,20)  
      enddo

      call profmean(cliss,imd,jmd,nm,trjm,yln,prof,xc,jt1,jt2,jtr,
     1     center,dyln, pte,coef,promax,jfm,profmm,km,profij,
     2     jadd)
c     --------  
      write(3,*)' end profmean'
      write(3,*)' 1938 cymx,cliss ',cymx(ic,jc,5),cliss(ic,jc,5)   ! B
c========================  calibration intensités:     cal(i,j,n)
c                                                          Y X
c              cymx(i,j,n)  +  profmm(k)  ->  cal(i,j,n)

c      -------------------------------profmm
c     channel 9   1    jt2
c             8        jt1    jt2
c             7               jt1    jt2
c             ---------------------------
c             1                      jt1    jt2    jmd
c                                 1   40     82    123        
c     calib
c     channel 6                   1  jt1    jt2    jmd          +jeps

c     canal 1  plot 0 -> jt2   pour i=ic
c                   1 _> jt1
c     canal 2  plot 1 -> jt2
c                   2 -> jt1
c     ----------------------      
c     canal 9  plot 8 -> jt2
c                   9 -> jt1      
     
c              clissp=cliss(ic,jlis2,np)
c               cal(i,j,n)=cymx(i,j,n)/(clissp*coef(ic,np))
c     ---------
c               jlis=j+pte(ncurv)*(i-ic)
      write(3,*)' nm,imd,jmd,km,ic,ncurv,jdel,jadd,jtr,jt1 ',
     1     nm,imd,jmd,km,ic,ncurv,jdel,jadd,jtr,jt1
c             885 123 459  443 5   0    0   43  41
      ic=(1+imd)/2
      do n=1,nm
        do  i=1,imd
         jdel=pte(ncurv)*(i-ic)
           do j=1,jmd 
              jp=j+jdel
c
c   n=1            jmd   jt2  jt1   (1)                 k=1+ jmd-j
c     2                       jt2   jt1    (1)          k=1+ jmd-j +(n-1)*jtr
c     3                             jt2    jt1          k=1= jmd-j +(n-1)*jtr
            k=1+jmd-jp+(n-1)*jtr
                kp=k
                if(k.lt.1)then
                  kp=1
                endif
                if(k.gt.km)then
                  kp=km
               endif
            if(cymx(i,jp,n).lt.1.)cymx(i,jp,n)=1.
            cal(i,j,n)=cymx(i,jp,n)/profmm(kp)
c     ----------
c            write(3,*)' kcy,profmm(kp),cal ',kcy,profmm(kp),cal(i,j,n)
            enddo              !  i   885
         enddo                 !  j   123
      enddo                     ! n

c -------------------------------------  ecriture   cal
         i=400 !imd-1   !(imd+1)/2    
      do n=1,nm
        write(3,*)' n,i,jdel ',n,i,jdel       
        do j=1,123,20    !1,jmd,40
c         cl=cal(i,j,n)
         k=1+jadd+n*jtr-(j-jt1)
         if(k.lt.1)then
            k=1
         endif
         if(k.gt.km)then
            k=km
         endif
         procl=profmm(k)
         write(3,*)' cal  n,i,j,cymx,profmm,k,cal ',
     1              n,i,j,cymx(i,j,n),profmm(k),k,cal(i,j,n)
         enddo                     ! j
      enddo                     ! n
c----------------------edges of channels
      i=(imd+1)/2
      do n=1,9
         write(3,*)' cal edges n j=1,jmd ',cal(i,1,n),cal(i,jmd,n)
      enddo
c============================================   plot  cal
      call par1('   x1cal',1,ix1cal)
      call par1('   x2cal',1,ix2cal)
 110  format('Intensity calibration X=',i4)
c     titre(25:28)=ix1cal
      ndes=2
      do icdes=1,ndes
         if(icdes.eq.1)icd=ix1cal
         if(icdes.eq.2)icd=ix2cal

      call pgadvance
      call pgslw(3)
      call pgsch(2.5)
      call pgvport(0.2,0.9,0.2,0.8)
c      call pglabel(titre(1:28))
      xl1=0.
      xl2=float(nm)
      yl1=0.
      yl2=2.
      call pgwindow(xl1,xl2,yl1,yl2)
      call pgbox    ('bcgint',1.,0,'bcints',0.5,5)

      call par1('   y1cal',1,iy1cal)
      call par1('   y2cal',1,iy2cal)
      write(titre(1:20),'(a)')'C    Calibration  Y='
      if(icdes.eq.1)write(titre(21:24),'(i4)')iy1cal
      if(icdes.eq.2)write(titre(21:24),'(i4)')iy2cal
      call pglabel('Channels  Full X-range',' ',titre(1:24))  
         
c   write(titre(10:29),'(a)')'min. aver. profile ='
c      write(titre(30:36),'(f7.0)')xminpro
      
         xl=float(jmd)-1
      do n=1,nm
         i=icd
         do j=1,jmd
            x(j)=float(n-1)+float(j-1)/xl
            y(j)=cal(i,j,n)
         enddo
         call pgline(jmd,x,y)
 115  format('  calib.ps   jmd,(y(nd),nd=1,jmd,20)', i4,2X,10f8.4)
         write(3,115)jmd,(y(nd),nd=1,jmd,20)                         ! B
      enddo                     !  n
      enddo                     !  icdes
      call pgend
      
      write(3,*)'(1985)  plot calib.ps'
      call par1('  icalib',1,icalib)
      if(icalib.eq.1) call system('okular calib.ps &')
c*************************************************     plot cal.ps
      call par1('   kedgx',1,kedgx)
      call par1('   kedgy',1,kedgy)

      call pgbegin(0,'cal.ps/ps',nm,1)
          call par1('  icalbl',1,icalbl)
          call par1('  icalwh',1,icalwh)
          calbl=float(icalbl)/1000.
          calwh=float(icalwh)/1000.
         tr(1)=0. 
         tr(2)=1.
         tr(3)=0.
         tr(4)=0.
         tr(5)=0
         tr(6)=1.
c     if(isonocor.eq.1)goto 212
c      im1=kedgx+1
c      im2=imd-kedgx
c      jm1=kedgy+1
c     jm2=jmd-kedgy
      im1=1
      im2=imd
      jm1=1
      jm2=jmd      
      write(3,*)' cal(im2,1,1)imd,jmd,im1,im2,jm1,jm2,nm,calbl,calwh ',
     1        cal(im2,1,1),imd,jmd,im1,im2,jm1,jm2,nm,calbl,calwh       !  B
 
      write(3,*)' plotcal2 cal  ip j=1,61,jm2, nc'
      jp1=1
      jp2=61
      jp3=jm2
      nc=5
c      do ip=1,2
c         print *,ip,cal(ip,jp1,nc),cal(ip,jp2,nc),cal(ip,jp3,nc)
c      enddo
c     call plotcal2(2,cal,im1,im2,jm1,jm2,nm,calbl,calwh,tr)
      call plotcal2(1,cal,im1,im2,jm1,jm2,nm,calbl,calwh,tr)
      call par1('icalplot',1,icalplot)
      if(icalplot.eq.1)call system('okular cal.ps &')
c     *************************************  solar observations obsD1,obsD2
c                                                                   obs
c      do 20 n=1,nm
c         do i=1,imdes
c            do j=1,jmdes
c               ical=j+kedgx
c               jcal=i+kedgy
c               des(i,j)=cal(ical,jcal,n)
c            enddo
c         enddo         

      call solarobs(xr,yr,imd,jmd,nm,cal,sobs,sobstot)
c     in               out again all


c           write(3,*)' plotcal2 cal '
c      call plotcal2(2,cal,im1,im2,jm1,jm2,nm,calbl,calwh,tr)
c      call par1('icalplot',1,icalplot)
c      if(icalplot.eq.1)call system('okular cal.ps &')
      
 210  format(2i4,9f6.0)
      ic=(imd+1)/2
      jc=(jmd+1)/2
c      call par1('isonocor',1,isonocor)
c      write(3,210)ic,jc,(sobs(ic,jc,n),n=1,nm) !   sobs
      write(3,*)' solarobs sobs(ic,jc,n) ',(sobs(ic,jc,n),n=1,nm)   ! sobs

      call par1('   kedgx',1,kedgx)
      call par1('   kedgy',1,kedgy)
c     if(isonocor.eq.1)goto 212
      im1=kedgx+1
      im2=imd-kedgx
      jm1=kedgy+1
      jm2=jmd-kedgy
      write(3,*)' obsD1: im2 ',im2
      do n=1,nm
         do jp=jm1,jm2   !  1,jmd
            do ip=im1,im2
               den=cal(ip,jp,n)
               if(den.eq.0.)den=0.5
c               sobsD1(ip,jp,n)=sobs(ip,jp,n)     !   sobsD1
c               sobsD2(ip,jp,n)=sobs(ip,jp,n)/den     !  sobsD2  calib
c               sobs(ip,jp,n)=sobs(ip,jp,n)/den   !  correction cal
               sobsD2(ip,jp,n)=sobs(ip,jp,n)         !calibré
               sobsD1(ip,jp,n)=sobs(ip,jp,n)*den     ! non calibré
c               sobs(ip,jp,n)                         ! calibré       
            enddo
         enddo
      enddo
c************************ plot  obs.ps
         tr(1)=0. 
         tr(2)=1.
         tr(3)=0.
         tr(4)=0.
         tr(5)=0.
         tr(6)=1.
      
      call pgbegin(0,'obs.ps/ps',nm,1)
          call par1('  iobsbl',1,iobsbl)  !  500 oui
          call par1('  iobswh',1,iobswh)  ! 1800
          sobsbl=float(iobsbl)
          sobswh=float(iobswh)

      write(3,*)' obs:  obswh,obsbl ',sobswh,sobsbl
          call plotcal2(2,sobs,im1,im2,jm1,jm2,nm,sobsbl,sobswh,tr)
c                       in              in          
      call par1('    iobs',1,iobs)
      if(iobs.eq.1)call system('okular obs.ps &')           ! oui
c*********************************            plots obsD1.ps, obsD2.ps
      do n=1,nm
         do jp=jm1,jm2
            Xmean1=0.
            Xmean2=0.
            do ip=im1,im2
               Xmean1=Xmean1+sobsD1(ip,jp,n)
               Xmean2=Xmean1+sobsD2(ip,jp,n)
            enddo
              Xmean1=Xmean1/float(im2-im1+1)
              Xmean2=Xmean2/float(im2-im1+1)
              if(jp.eq.jc)write(3,*)' Xmean1 for jc:',Xmean1
            do ip=im1,im2
               sobsD1(ip,jp,n)=1000.*sobsD1(ip,jp,n)/Xmean1  !-Xmean1
               sobsD2(ip,jp,n)=1000.*sobsD2(ip,jp,n)/Xmean2  !-Xmean2
            enddo
         enddo
         write(3,*)' n, sobsD1(ic,jc,n)=',n,sobsD1(ic,jc,n)
         write(3,*)' n, sobsD2(ic,jc,n)=',n,sobsD2(ic,jc,n)
      enddo
c***************                                   obsD1.ps
      call pgbegin(0,'obsD1.ps/ps',nm,1)
          call par1(' iobsDwh',1,iobsDwh)
          call par1(' iobsDbl',1,iobsDbl)
          sobsDwh=float(iobsDwh)
          sobsDbl=float(iobsDbl)
     
      write(3,*)' obsD1.ps: wwh,bbl ',iobsDwh,sobsDbl
          print *,' sobsD2.ps:  '
          call plotcal2(2,sobsD1,im1,im2,jm1,jm2,nm,sobsDbl,sobsDwh,tr)
c                       in              in          
      call par1('  iobsD1',1,iobsD1)
      if(iobsD1.eq.1)call system('okular obsD1.ps &')
c***************************************                 obsD2.ps  calib
           call pgbegin(0,'obsD2.ps/ps',nm,1)
     
      write(3,*)' obsD2.ps: wwh,bbl ',iobsDwh,sobsDbl
          print *,' sobsD2.ps:  '
          call plotcal2(2,sobsD2,im1,im2,jm1,jm2,nm,sobsDbl,sobsDwh,tr)
c                       in              in          
      call par1('  iobsD2',1,iobsD2)
      if(iobsD2.eq.1)call system('okular obsD2.ps &')

      return
      end
c                fin calib      
c=====================================================  
      subroutine plot_line(trjm1,cymx,imd,jmd,i1,i2,iliss,jliss,
     1                     il1,il2,jl1,jl2,nm,
     2                     cliss,trjm,yln,center,pte,jtr)
c                ---------        out liss droite droite
c     1                              fna,x,y,
c                                       ecomonie        
c     2              il1,il2,jl1,jl2,iil1,iil2,iliss,jmarge,jparli,
c     3         tabp,tabq,nw,win,milsec, curv,cr,yln,ngood,ldes,
c                                            out
c     4         dyln,nq,idy,trjm,nw1,nw2)
      dimension cymx(2000,200,24)
      dimension cliss(2000,200,24)
c      character*22 fna
      character*33 titre   

c      character*18 titre2
      dimension x(2000),y(2000),tabp(200),tabq(200),yln(2000,24),
     1     center(2000,24),cline(2000),tj(24),pte(24),
     2     zln(2000,24)
c      call par1('   marge',1,marge)
      write(3,*)' entree dans plot_line'
      ic=(1+imd)/2
      jc=(1+jmd)/2
      xl=float(jmd)-1.
      yl=float(imd)-1.
c---------------------------------------------------
      write(3,*)' 2319 cymx, cliss (ic,jc,5) ',
     1                 cymx(ic,jc,5), cliss(ic,jc,5)      !  B)
c--------------------------------------------------
c     npiv=nm/2-1
c      call par1('    curv',1,kcurv)
c      call par1('  ncurv1',1,ncurv1)
c      call par1('  ncurv2',1,ncurv2)
      call par1('      nr',1,nr)
      ncurv=nr
      ncurv1=nr
      ncurv2=nr+1
      nc=(nm+1)/2
c----------------------------
      do n=nc-1,nc+1
      do i=1,imd
         yln(i,n)=0.
      enddo
      enddo
      
      do n=ncurv1,ncurv2  !ncurv-1,ncurv2  
      call linecurv(cymx,imd,jmd,il1,il2,jl1,jl2,nm,n,iliss,jliss,cliss,
c          -------- in                      in      *
     1         ncurv1,ncurv2,ncurv, yln,zln,xa,xb,ylna,ylnb,center,pte)
c                                   out                      *     *
      enddo
      
      write(3,*)' 2342 cymx,cliss ',cymx(ic,jc,5),cliss(ic,jc,5)   ! B
c-------------------------------    A Channels: distance between  line centers
        n1=ncurv1-1   !nc-1
        n2=ncurv1  
        n3=ncurv2
c----------------------------------  plot 3 profiles supprimé       
c---------------------------------------------             plots  nc,nc+1
      do n=ncurv1,ncurv2
       write(titre(1:20),'(a)')'A   Line center  n= '
       write(titre(21:22),'(i2)')n
       write(titre(23:33),'(a)')'           '
      call pgslw(3)
      call pgadvance
      call pgsch(2.5)  
      call pgvport(0.2,0.9,0.2,0.8) !       0.1,0.9,0.1,0.7)   !0.8
      call pglabel('X','Y',titre)   !  J, I
      call pgwindow(0.,xl,0.,yl)
      call pgbox('bcnts',50.,5,'bcnts',200.,2)
c      call pglabel('J','I',//titre)
          print *,' n= ',n
      
          ilm=il2-il1+1
          print *,' il1,il2 ',il1,il2
c      print *,' yln(i,n) ',(yln(i,n),i=il1,il2,100)
c      print *,' center   ',(center(i,n),i=il1,il2,100)
      do i=il1,il2
      x(i-il1+1)=yln(i,n)  !yln(i,n)       !  à partir de 0  pour il1   (liss)
c                *****
      y(i-il1+1)=i-1
      enddo
 1    format(' il1,il2,x ',2i6,10f6.0)
 2    format(' il1,il2,y ',2i6,10f6.0)
      write(3,1)il1,il2,(x(i),i=1,ilm,100)
      write(3,2)il1,il2,(y(i),i=1,ilm,100)
      call pgslw(4)
      call pgline(ilm,x,y)      !    raie

c                                          center (all points from 1 to imd) 

c      print *,' n,center ',n,(center(i,n),i=1,imd,100)
c      write(3,*)' n,center ',n,(center(i,n),i=1,imd,100)
      do i=1,imd
         y(i)=i-1
         x(i)=center(i,n)                              !  gche puis droite
c             ------
      enddo
      call pgslw(3)
c      call pgsls(1)
      call pgline(imd,x,y)
      call pgslw(1)
      call pgsls(1)
c      x(1)=0.  !1.
c      x(2)=xl
c      y(1)=yl/2.
c
c      y(2)=y(1)
c      call pgsls(2)
c      call pgline(2,x,y)         !         axe milieu
c          ------
c      call pgsls(1)
      enddo                     !  n                 channels
c************************************************************
c                                  trjm < Jtrans   = translation n, n+1
c      write(95,*)' '
      print *,' center(ic,ncurv1),center(ic,ncurv2)',
     1        center(ic,ncurv1),center(ic,ncurv2)  
 3    format(f6.1,i6)
      trjm=(center(ic,ncurv2)-center(ic,ncurv1))/float(ncurv2-ncurv1)
      jtrprox=trjm+0.5

      call trans(cliss,imd,jmd,nm,jtrprox,jtrcalc,devcalc)
      write(3,*)' 2342 cymx,cliss ',cymx(ic,jc,5),cliss(ic,jc,5) ! B
C------------------------------------------
      write(3,*)' 2419 cymx,cliss ',cymx(ic,jc,5),cliss(ic,jc,5) ! B
      write(3,*)' trans: jtrprox,jtrcalc,devcalc ',
     c                   jtrprox,jtrcalc,devcalc
c     -----   translation of wavelengths between channels
      call par1('  ntrans',1,ntrans)
      if(ntrans.eq.0)then
         jtr=jtrcalc
         trjm=jtr
      else
         trjm=trjm1
         jtr=trjm+0.5
      endif

      dev=devcalc
 4    format(' Translations n,n+1  jtr,dev ',i3,f6.3)
      print 4,jtr,dev         
      write(3,4)jtr,dev
c-----------------------------
c                                B Same wavelength channels n,n+1
            call pgslw(3)
      call pgsch(2.5)
      call pgadvance
      call pgvport(0.2,0.9,0.2,0.8)
c     call pglabel('X','Y','B  Same wavelength, channels n,n+1')
      call pglabel('X','Y','B  Wavelength range used for mean profile')
      xl=float(jmd)-1.
      call pgwindow(0.,xl,0.,yl)
      call pgbox('bcnts',50.,5,'bcnts',200.,2)

        n0=nc   !(ncurv1+ncurv2)/2      pte prise sur nc
      do i=1,imd
         cline(i)=center(i,n0)-center(ic,n0)
      enddo
      print *,'cline ',(cline(i),i=1,imd,200)
      do n=1,2
         xc=xl/2.+(2*n-3)*0.5*float(jtr)        !trjm/2.
c                              -------     positions entre pixels
         print *,'xc',xc
         do i=1,imd
            y(i)=i-1.
            x(i)=xc+cline(i)
         enddo
         print *,'x',(x(i),i=1,imd,200)
         print *,'y',(y(i),i=1,imd,200)
         call pgline(imd,x,y)           !  Repères de landas dans channels
c         xt=xc+2.
c         yt=yl/2.+200
c         if(n.eq.1)call pgtext(xt,yt,'n')
c         if(n.eq.2)call pgtext(xt,yt,'n+1')
      enddo 
      
c     tracé de Ts
      write(3,*)'plot Ts'
      trj2=trjm/2.
      x(1)=xl/2.-trj2
      x(2)=xl/2.+trj2
      y(1)=yl/2.
      y(2)=y(1)
      write(3,*)' trj2 ',trj2,x(1),x(2),y(1),y(2)
      call pgsls(2)
      call pgline(2,x,y)
      call pgsls(1)

      xx=xl/2.-10
      yy=yl/2.+50
      ktrj=trjm+0.5
      write(titre(1:5),'(a)')'Ts = '
      write(titre(6:8),'(i3)')ktrj
      call pgslw(3)
      call pgtext(xx,yy,titre(1:8))
      yy=yl/2.
      xx=xl/2.-trj2-9.
      call pgtext(xx,yy,'W1')
      xx=xl/2.+trj2+2.
      call pgtext(xx,yy,'W2')
      yy=yl/5.
      xx=32.
      call pgtext(xx,yy,'L1')
      xx=88.
      call pgtext(xx,yy,'L2')
      return
      end
c---------------------------------------
      subroutine linecurv(cymx,imd,jmd,il1,il2,jl1,jl2,nm,n,iliss,jliss,
c                                                         *      
     1    cliss,ncurv1,ncurv2,ncurv,yln,zln,xa,xb,ylna,ylnb,center,pte)
c                                   out  à partir de 0                
      real*4 cymx(2000,200,24)
      dimension tabp(200),tabq(200),yln(2000,24),center(2000,24),
     1  pte(24),zln(2000,24),x0(200),z0(200),yic(24),cliss(2000,200,24),
     2  dyln(200,24),xparab(2000),yparab(2000),pcoef(10),pp(2000)
c                               ?      ?         cmin(2000,24)
      dimension xpar(2000),ypar(2000),pds(2000),coefpar(10)
            DOUBLE PRECISION C,F,PIVOT
c      DIMENSION X(2000),Y(2000),P(2000),COEF(10),F(10),C(11,11)
      call par1('  jparab',1,jparab)
      jl1=jl1+jparab
      jl2=jl2-jparab
      print *,' n entree linecurv  il1,il2,jl1,jl2 ',n,il1,il2,jl1,jl2
      write(3,*)' '

      write(3,*)' 2494 debut linecurv n= ',n
      print *,' n,il1,il2,jl1,jl2 ',n,il1,il2,jl1,jl2
      write(3,*)' n,il1,il2,jl1,jl2 ',n,il1,il2,jl1,jl2
      ic=(il1+il2)/2
 12   format(' 2585 linecurv cliss ',20f6.1)     !  B
      print *,' cliss(ic,j,n),j=1,jmd,10 '
      print 12,(cliss(ic,j,n),j=1,jmd,10)
      write(3,*)' cliss(ic,j,n),j=1,jmd,10 '
      write(3,12)(cliss(ic,j,n),j=1,jmd,10)
c********************************         il1,il2
      do i=il1,il2
         piv=cliss(i,jl1,n)     !+cliss(i,jc,n-1)+cliss(i,jc,n+1)
c     piv=cymx(i,jl1,n)
c--------------------------------          jl1,jl2       
         do j=jl1,jl2          ! -1 ?
          piv1=cliss(i,j,n)    !+cliss(i,j,n-1)+cliss(i,j,n+1)
c             piv=cymx(i,j,n)
              piv=amin1(piv,piv1)
c                 -----
         enddo
            japprox=jl1
         do j=jl1,jl2
           piv2=cliss(i,j,n)    !+cliss(i,j,n-1)+cliss(i,j,n+1)
c            piv2=cymx(i,j,n) 
            if(piv2.eq.piv)then
               japprox=j
            endif
         enddo                  !   jl1,jl2
c------         
         yln(i,n)=japprox
         zln(i,n)=cliss(i,japprox,n)
c------
c      if((i-il1)*(i-il2).eq.0)then   

c     call par1('  jparab',1,jparab) 
      ndpar=1+2*jparab
c         xpar(jparab+1)=japprox
c      do nd=1,jparab
c         xpar(nd)=japprox-nd
c         xpar(nd+jparab)=japprox+nd
c      enddo
      do nd=1,ndpar
         pds(nd)=1.
      enddo

      do nd=1,ndpar
         japp=japprox-jparab+nd-1
         xpar(nd)=japprox-jparab+nd-1
         ypar(nd)=cliss(i,japp,n)
      enddo
      call DPMCAR(xpar,ypar,pds,ndpar,3,coefpar)
      yln(i,n)=-coefpar(2)/(2.*coefpar(3))          !   yln
        
c      write(3,*)' parab: xpar ',(xpar(nd),nd=1,ndpar)
c     write(3,*)' parab: ypar ',(ypar(nd),nd=1,ndpar)
c      write(3,*)' parab: coefpar ',(coefpar(ndco),ndco=1,3)
c      write(3,*)' parab: japprox yln(i,n) ',japprox,yln(i,n)
c      endif
c-----------                                 endif  il1 ou il2
      enddo                     !     il1,il2
      
 15   format(' 2603 n,yln(i,ncurv) ',10f5.0)
 16   format(' 2603 n,zln(i,ncurv) ',10f5.0)
      write(3,*)' linecurv: yln(i,ncurv)i)il1,il2,50   n,i ',n,i  
      write(3,15)(yln(i,ncurv),i=il1,il2,50) ! N
      write(3,*)' linecurv: zln(i,ncurv)i)il1,il2,50   n,i ',n,i 
      write(3,16)(zln(i,ncurv),i=il1,il2,50)

c*****************************
      call par1(' leastsq',1,leastsq)
      if(leastsq.eq.1)then                                !   leastsq
         
      ipar=il2-il1+1
      do il=1,ipar
         i=il1+il-1
         pds(il)=1.
         xpar(il)=float(i)
         ypar(il)=yln(i,n)
c                  *  
      enddo
      write(3,*)' least squares: il1,il2, xpar, ypar (1-2)',
     1            il1,il2,xpar(1),xpar(ipar),ypar(1),ypar(ipar) 
      call DPMCAR(xpar,ypar,pds,ipar,2,coefpar)
      do i=1,imd
         center(i,n)=coefpar(1)+float(i)*coefpar(2)
c           *
      enddo
        ic=(il1+il2)/2
      yic(n)=center(ic,n)
      pte(n)=coefpar(2)
c      *
      write(3,*)' least squares coefpar(1), coefpar(2) ',
     1     coefpar(1),coefpar(2)
      write(3,*)' n, center(1,imd,100) ',n,(center(i,n),i=1,imd,100)
c------------------------------
c     if((n-ncurv1)*(n-ncurv2)*(n-ncurv)=0)then
      
c**********************************************
      else                                                   ! 2  moyennes
c     center         2 intervalles                 center
c     center lissé
c     calculs avec yln entre          i1 et ic-1      ic et i2
c             moyennes                  ylna            ylnb
c             moy affectées à            xa              xb
      ylna=0.
      ylnb=0.
      den=0.
      ic=(il1+il2)/2
      ic1=ic-1
      do i=il1,ic1
         ylna=ylna+yln(i,n)
         den=den+1.
      enddo
         ylna=ylna/den
         xa=float(ic+il1)/2.-1.
         
      den=0.
      do i=ic,il2
         ylnb=ylnb+yln(i,n)
         den=den+1.
      enddo
         ylnb=ylnb/den
         xb=float(ic+il2)/2.-1.
         
 3    format(' 1, i1,ic-1, ic,i2, imd,  xa,xb,ylna,ylnb ')
 4    format(6i6,4x,4f6.1)
      write(3,3)
      ipiv=1
      write(3,4)ipiv, il1,ic1, ic,il2, imd,  xa,xb,ylna,ylnb  
 5    format('linecurv:  n   center ',i4,100f6.1)
      den=il2-il1
      call par1('   ncurv',1,ncurv)
      do i=1,imd
         center(i,n)=ylna+(ylnb-ylna)*(float(i-il1)/den) ! à partitr de 0
c     yln(i,n)=center(i,n)
         if(n.eq.ncurv)dyln(i,n)=center(i,nc)-center(ic,nc)
c      write(3,*)' refer linecurv: center(i,n,n=1,nm) ',
c     1                           (center(i,n),n=1,nm)
c      print *,' linecurv: center(i,n,n=1,nm) ',(center(i,n),n=1,nm)
c         yln(i,n)=center(i,n)    non yln avant moy pour nc
      enddo
      yic(ncurv)=center(ic,ncurv)
      pte(ncurv)=(ylnb-ylna)/(xb-xa)       !  slope line/i
c      write(3,5)n,(center(i,n),i=1,imd,100)
c      print 5,n,(center(i,n),i=1,imd,100)
c     print *,' slope  ',pte(n)
      endif
c*************************************************      
 8    format(' n, yic, pte dy/dx ',i4, f6.1, f8.5)
      print 8,n,yic(n),pte(n)
      write(3,8)n,yic(n),pte(n)
      print *,' cymx de 1 à 10 ',(cymx(ic,j,n),j=1,10)
c      ic=(1+nm)/2

      return
      end
c-------------------------------------------------------
      subroutine profmean(cliss,imd,jmd,nm,trj,yln,prof,xc,jt1,jt2,jtr,
     1     center,dyln,pte,coef,promax,jfm,profmm,km,profij,jadd)
c     -------         out
      dimension yln(2000,24),prof(2000,100,24),profmm(2000),dyln(200,24)
      dimension pronoc(2000)   !  profile no coeff
      dimension j1tr(24),j2tr(24),x(1000),y(1000),coef(2000,24),xk(2000)
      dimension cliss(2000,200,24),zln(2000,24),pte(24),center(2000,24),
     1     profij(2000,200,24),profm(2000,24),coeff(24),y1(100),y2(100),
     2     yk(2000)
      character*34 titre
      xjtr=float(jt2-jt1)
c        call par1('  jtrans',1,jtrans)
c        if(jtrans.ne.0)jtr=jtrans
c      call par1('   ncurv',1,ncurv)
      call par1('      nr',1,nr)
      ncurv=nr
c        kincli=(center(imd,ncurv)-center(1,ncurv))/2.+1
      print *,' profmm  center(1,ncurv),center(imd,ncurv),jtr,kincli',
     1                  center(1,ncurv),center(imd,ncurv),jtr,kincli
c      call par1('jaddprof',1,jadd)
c      kadd1=jmd-jt2
c      kadd2=jt1-1
      write(3,*)' profmean: jt1,jt2,jtr,jadd ',jt1,jt2,jtr,jadd ! ?
      jpm=jtr+1
      ic=imd/2+1
      jc=jmd/2+1
      print *,' profmean'
c      print *,' jtr,kadd1,kadd2,km,jpm,ic,jc',
c     1           jtr,kadd1,kadd2,km,jpm,ic,jc
c     profmm = mean profile from first to last channel n
c-------------------------------------------------------------
      write(3,*)' 2622  cliss(ic,jc,5) ', cliss(ic,jc,5)
      promax=0.
c 1    format(' jtr,jc,ic,nc,jt1,jt2 ',6i4) 
c      write(3,1) jtr,jc,ic,nc,jt1,jt2
c      print 1,jtr,jc,ic,nc,jt1,jt2,jpm
c-----------------------
       write(3,*)' nm,imd,jpm,jt1,jt2 ',nm,imd,jpm,jt1,jt2
c========================
      write(3,*)' new '
c     calcul de profm(jp,n)  et  coeff(n)
      promax=0.
      do n=1,nm
        do j=1,jmd             !jt1,jt2           !! abscisse dans prof
c            jcl=jt2-jp+1                 !  inversion j
             profm(j,n)=cliss(ic,j,n)
c                        ----          -----
              promax=amax1(promax,profm(j,n))
           enddo                ! j
         write(3,*)' promax ',promax
         write(3,*)' profmean profm n  jc,n=1,61,jmd ',
     1        profm(1,n),profm(61,n),profm(jmd,n)
      enddo    ! n
c----------------------------------------------
      call par1('      nr',1,nr)
      ncurv=nr
      print *,' ic,nm,ncurv ',ic,nm,ncurv
      coeff(1)=1.
      do n=2,n
         coeff(n)=coeff(n-1)*profm(jt2,n)/profm(jt1,n-1)
         print *,' n,profm(jt2,n),profm(jt1,n-1),coeff(n)',
     1               n,profm(jt2,n),profm(jt1,n-1),coeff(n)

      enddo
        coeff0=coeff(nr)
      do n=1,nm
         coeff(n)=coeff(n)/coeff0
      enddo
c dev
            dev=0.
      do n=1,nm
         piv=abs(coeff(n)-1.)
         dev=amax1(dev,piv)
      enddo
 8    format(' jtr,jt1,jt2,dev,coeff(n) ',3i4,f6.3,2x,24f6.3)
      print 8,jtr,jt1,jt2,dev,(coeff(n),n=1,nm)
      write(3,8)jtr,jt1,jt2,dev,(coeff(n),n=1,nm)
c=============================
c ----------- plot 1 2 3 ... 9
      xjtr=float(jtr)
c------------------------------------------------plot profmean  1,2,3,   9
c      call par1('jaddprof',1,jadd)
      call pgadvance
      call pgslw(3)
      call pgsch(2.5)
      call pgvport(0.2,0.9,0.2,0.8)
      call pglabel('Channels with inverted Ts ranges','Intensity',
     1             'B  Mean profile')
c--------------------------------------------pofmm  2605
c----------------------mean profile
c     jmd  jt2  jt1   1
c               jt2   jt1
c                     jt2   jt1
c                      ------           
c                           jt2   jt1    1 
c 
      do 40 n=1,nm              !    9
         if(n.eq.1)then         !  channel 1
         j1=jmd
         j2=jt1
         lm=jmd-jt1+1
         k1=1
         k2=lm
         k3=k2
         write(3,*)' n,j1,j2,lm,k1,k2,k3 ',n,j1,j2,lm,k1,k2,k3
         do k=1,lm
            j=j1+1-k
            pronoc(k)=profm(j,n)
            profmm(k)=pronoc(k)/coeff(n)                 
         enddo
         endif
         if(n.ge.2.and.n.le.(nm-1))then
            j1=jt2
            j2=jt1
            lm=j1-j2+1
            k1=k3
            k2=k1+jt2-jt1
            k3=k2
            write(3,*)' n,j1,j2,lm,k1,k2,k3 ',n,j1,j2,lm,k1,k2,k3
         do k=k1,k2
            j=j1-(k-k1)
            pronoc(k)=profm(j,n)
            profmm(k)=pronoc(k)/coeff(n)                 
         enddo
         endif
         if(n.eq.nm)then
             j1=jt2
             j2=1
             lm=jt2-1
             k1=k3
             k2=k1+lm
             k3=k2
             write(3,*)' n,j1,j2,lm,k1,k2,k3 ',n,j1,j2,lm,k1,k2,k3
         do k=k1,k2
             j=j1-(k-k1)
             pronoc(k)=profm(j,n)
             profmm(k)=pronoc(k)/coeff(n)                 
          enddo
          endif
          km=k3
 40      continue
 23      format(10f6.0)
         write(3,*)' profmean pronoc  first 40 ',km
         write(3,23)(pronoc(k),k=1,40)
      write(3,*)' profmean profmm km ',km
      write(3,23)(profmm(k),k=1,km,10),profmm(km)
 26   format(' profmm km-10,km ',11f6.0)
      write(3,26)(profmm(k),k=km-10,km)
c--------------------------------plot pronoc,profmm
c      new2604e
c      xl1=-xadds               
c      xl2=float(nm)+xadds            !                     
c      yl=promax*1.5
c      call pgwindow(xl1,xl2,0.,yl)
c      call pgbox('bcgint',1.,0,'bcints',1000.,5) !   1 2 3 ...9
      xjtr=float(jt2-jt1)
      xl1=-float(jmd-jt2)/xjtr
      xl2=float(nm)+float(jt1-1)/xjtr                   !
c     xl2=(float(km-1)+float(jt1-1))/xjtr        !                     
      call pgwindow(xl1,xl2,0.,2500.)
      call pgbox('bcgint',1.,0,'bcints',1000.,5) !   1 2 3 ...9
      do k=1,km
      xk(k)=xl1+float(k-1)/xjtr
      enddo
      call pgslw(1)
      call pgsls(1)
      do k=1,km
         yk(k)=pronoc(k)
      enddo
      call pgline(km,xk,yk)
      call pgslw(3)
      call pgsls(1)
      call pgline(km,xk,profmm)
      return   
      end
c=====================================
      subroutine transpec(ntrans,xr,yr,imd,jmd,nm,trj)
      dimension xr(24,3,2),yr(24,3,2)
      jm=jmd
      do n=1,nm
         write(3,*)' xr(n,1,1),xr(n,3,2),yr(n,1,1),yr(n,3,2)  ',
     1               xr(n,1,1),xr(n,3,2),yr(n,1,1),yr(n,3,2)
      enddo
      call par1('  mupris',1,mupris)
      call par1('  mustep',1,mustep)
      nc=ntrans
      chajb=0.5*(yr(nc,1,2)-yr(nc,1,1)+yr(nc,3,2)-yr(nc,3,1))
c                      D          A          F          C         
      vecjb=0.125*(yr(nc+1,1,1)+yr(nc+1,3,1)+yr(nc+1,1,2)+yr(nc+1,3,2)
     1            -yr(nc-1,1,1)-yr(nc-1,3,1)-yr(nc-1,1,2)-yr(nc-1,3,2))
c                          A            C            D            F
      write(3,*)' transpec: ntrans,chajb,vecjb,jm ',
     1                      ntrans,chajb,vecjb,jm
c
      write(3,*)' ntrans,jm,nm,mustep,vecjb,mupris,chajb ',
     1     ntrans,jm,nm,mustep,vecjb,mupris,chajb
        trj=float(jm-1)*mustep*vecjb/(float(mupris)*chajb)
      write (3,*)'  trj: trj,xr(ntrans,1,1) ',trj,xr(ntrans,1,1)
              
      return
      end
c-----------------------------------------------------
c===========================================================
      subroutine trans(cliss,imd,jmd,nm,jtrprox,jtrcalc,devcalc)
      dimension cliss(2000,200,24),co(24,50),dev(24,50),cl(2,24,50),
     1          devm(50),jtrans(50)
c                    kt

      print *,' call trans '
      write(3,*)' call trans '
      jtrans1=jtrprox-10
      jtrans2=jtrprox+10
      ic=(1+imd)/2
      jc=(1+jmd/2)
      nc=(1+nm)/2

      ktm=jtrans2-jtrans1+1

      do kt=1,ktm
      jtrans(kt)=jtrans1+kt-1
      jt1=jc-jtrans(kt)/2
      jt2=jt1+jtrans(kt)             ! even jt odd
      jk1=jmd-(jt1-1)    
      jk2=jmd-(jt2-1)
c      print *,' kt,jtrans,jmd,jc,jt1,jt2,jk1,jk2 ',
c     1          kt,jtrans,jmd,jc,jt1,jt2,jk1,jk2
           co(1,kt)=1.
         do n=1,nm
            cl(1,n,kt)=cliss(ic,jk1,n)
            cl(2,n,kt)=cliss(ic,jk2,n)
         enddo
c         print *,' kt,cliss(ic,jk1,1),cliss(ic,jk2,nm) ',
c     1             kt,cliss(ic,jk1,1),cliss(ic,jk2,nm)
         do n=2,nm
            co(n,kt)=co(n-1,kt)*cl(2,n-1,kt)/cl(1,n,kt)
         enddo
         cpiv=co(nc,kt)
         do n=1,nm
            co(n,kt)=co(n,kt)/cpiv
            dev(n,kt)=co(n,kt)-1.
         enddo

 1       format(' jtrans, co(n,kt),n=1,nm) ',i3,24f6.3)
         print 1,jtrans(kt),(co(n,kt),n=1,nm)
 2       format(' jtrans, dev(n,kt),n=1,nm) ',i3,24f7.3)
         print 2,jtrans(kt),(dev(n,kt),n=1,nm)
         print *,' '
         write(3,2)jtrans(kt),(dev(n,kt),n=1,nm)
         write(3,*)' '
            devm(kt)=0.
         do n=1,nm
            dev(n,kt)=abs(co(n,kt)-1.)
            devm(kt)=amax1(devm(kt),dev(n,kt))
         enddo
      enddo                     ! kt
      
         devmin=1.
         do kt=1,ktm
            devmin=amin1(devm(kt),devmin)
         enddo
         do kt=1,ktm
            if(devm(kt).eq.devmin)then
               jtrcalc=jtrans(kt)
               devcalc=devmin
               goto 10
            endif
         enddo
 10   continue
      
      return
      end
c===========================================
      SUBROUTINE parafit(Y,I1,I2,L,Z)
      DIMENSION Y(1),Z(1)
        IF(L.EQ.0)THEN
          DO I=I1,I2
          Z(I)=Y(I)
          ENDDO
        RETURN
        ENDIF
C
        IF(L.GE.10000)THEN
        X=0.
          DO I=I1,I2
          X=X+Y(I)
          ENDDO
        X=X/FLOAT(I2-I1+1)
          DO I=I1,I2
          Z(I)=X
          ENDDO
        RETURN
        ENDIF
C
      SX2=0.
      SX4=0.
        DO N=1,L
        N2=N*N
        SX2=SX2+N2
        SX4=SX4+N2*N2
        ENDDO
        SX2=2*SX2
        SX4=2*SX4
      D=(2*L+1)*SX4-SX2**2
      IA=I1+L
      IB=I2-L
      IF(IB.LT.IA)RETURN
C
      DO100 I=IA,IB
      SY=0.
      SYX=0.
      SYX2=0.
      IL1=I-L
      IL2=I+L
        DO IP=IL1,IL2
        DI=IP-I
        SY=SY+Y(IP)
        SYX=SYX+Y(IP)*DI
        SYX2=SYX2+Y(IP)*DI*DI
        ENDDO
      A=(SY*SX4-SYX2*SX2)/D
      Z(I)=A
C
      IF(I.EQ.IA.OR.I.EQ.IB)THEN
      B=SYX/SX2
      C=((2*L+1)*SYX2-SY*SX2)/D
      ENDIF
C
      IF(I.EQ.IA)THEN
        DO IP=I1,IA
        DI=IP-IA
        Z(IP)=A+DI*(B+DI*C)
        ENDDO
      ENDIF
C
      IF(I.EQ.IB)THEN
        DO IP=IB,I2
        DI=IP-IB
        Z(IP)=A+DI*(B+DI*C)
        ENDDO
      ENDIF
C
100   CONTINUE
      RETURN
      END
c*********************************************
      subroutine plotcal2(nplot,cal2,im1,im2,jm1,jm2,nm,bbl,wwh,tr)
c                                        in  in
      dimension des(200,2000),cal2(2000,200,24),tr(6)
c                  ides  jdes      ical jcal
      write(3,*)' plotcal2 cal(im2,1,1) im1,im2,jm1,jm2,nm,bbl,wwh',
     1           cal2(im2,1,1), im1,im2,jm1,jm2,nm,bbl,wwh
      ic=(im1+1)/2
      jc=(jm1+1)/2

      call par1('   kedgx',1,kedgx)
      call par1('   kedgy',1,kedgy)
      write(3,*)' kedgx,kedgy ',kedgx,kedgy
c      write(3,*)(' plotcal2: nplot  imd,jmd,nm  ic,jc  bbl,wwh')
c 1    format(8x,6i5,2f10.3)
c      write(3,1)  nplot, imd,jmd,nm, ic,jc, bbl,wwh
c 2    format(' n, calplot(ic,jc)',i4,2x,f8.3)
c     do n=1,nm
c         write(3,2)n,cal(ic,jc,n)
c      enddo
c      if(wwh.eq.0.)then     !    wwh=0.
c          bbl=10000.
c          wwh=0.
c      do n=1,nm
c          do j=1+kedgx,jmd-kedgx
c             do i=1+kedgy,imd-kedgy
c                piv=cal(i,j,n)
c                bbl=amin1(piv,bbl)
c                wwh=amax1(piv,wwh)
c             enddo
c          enddo
c       enddo
c      else                    !  wwh.ne.0.
c         if(nplot.eq.1)then
c            call par1('  icalwh',1,icalwh)
c            wwh=float(icalwh)/1000.
c            call par1('  icalbl',1,icalbl)
c            bbl=float(icalbl)/1000.
c         else
c            call par1('  iobswh',1,iobswh)
c            call par1('  iobsbl',1,iobsbl)
c            wwh=float(iobswh)
c            bbl=float(iobsbl)
c         endif
c      endif

      imdes=jm2-jm1+1    !  permutation
      jmdes=im2-im1+1
c      write(3,*)' plotcal2:  bbl,wwh, imdes,jmdes',bbl,wwh,imdes,jmdes 

      do 20 n=1,nm
         do i=1,imdes
            do j=1,jmdes
               icaldes=j+kedgx
               jcaldes=i+kedgy
               des(i,j)=cal2(icaldes,jcaldes,n)
            enddo
         enddo         
c        do jdes=1,imdx        !  123  y des   x cal   kedgx
c        do ipdes=1,jmdy     !  885  y des   x cal   kedgy

c 12    format (' plotcal2: n,imdes,jmdes,des(60,400)',3i5,f10.4) 
c           write(3,12)n,imdes,jmdes,des(60,400)
         call pgadvance
         call pgvport(0.25,0.75,0.25,0.8)   ! 0.3,0.7
         xi1=0.
         xi2=imdes-1
         yj1=0.
         yj2=jmdes-1
        tr(1)=0. 
        tr(2)=1.
        tr(3)=0.
        tr(4)=0.
        tr(5)=0.
        tr(6)=1.
        
      call pgwindow(xi1,xi2,yj1,yj2)
      call pgsch(5.)
      call pgslw(2)
      call pgbox('bcints',100.,2,'bcints',100.,2)
c      if(kyb.eq.1)call pglabel(' ','pixels ',' ')
c      if(kyb.eq.2)call pglabel(' ','pixels ',' ')
c      call pgline(5,xdes,ydes)
c      bbl=0.5
c      wwh=1.5
      write(3,*)' PGGRAY: nplot,n,bbl,wwh,des(x=50,y=400) ',
     1                    nplot,n,bbl,wwh,des(51,401)
      call PGGRAY(des,200,2000,jm1,jm2,im1,im2,bbl,wwh,tr)
 20   continue                          !      400 1900
      call pgend
      return
      end
c===========================
      subroutine map3(cymx,im,jm,nm)
      dimension cymx(2000,200,24)
      dimension tab(1536,1536),tr(6),xdes(5),ydes(5)
      character*7 cfile
      character*10 cfileps
c
      write(3,*)' Entree map3'
      bl=0.      !300.   ! black
      wh=0.                     !700. ! white
      cfileps(1:7)=cfile(1:7)
      cfileps(8:10)='.ps'
c---------------------------------------------------------
        tr(1)=0. 
        tr(2)=1.
        tr(3)=0.
        tr(4)=0.
        tr(5)=0.
        tr(6)=1.
c-------------------------------
      kpiv1=1
      kpiv2=1
      if(nm.gt.1)then
         kpiv1=1
         kpiv2=9
      endif
c     call system('rm //cfileps//')
c     call par1('  iflat1',1,iflat1)
      
c     call par1('  iflat2',1,iflat2)
      iflat2=0
      if(iflat2.eq.1) then
c      call pgbegin(0,'flat2.ps/ps',kpiv1,kpiv2)
c      if(iflat2.eq.2) call pgbegin(0,'Obs.ps/ps',kpiv1,kpiv2)
c--------------------------------min max
          bbl=bl
          wwh=wh
       if(wwh.eq.0.)then
          bbl=10000.
          wwh=0.
          do n=1,nm
          do j=1,jm
             do i=1,im
                piv=cymx(i,j,n)
                bbl=amin1(piv,bbl)
                wwh=amax1(piv,wwh)
             enddo
          enddo
       enddo
          print *,' l= ',l,'   bbl=',bbl,'   wwh=',wwh
       endif
c-------------
       do n=nm,1,-1
         do j=1,jm
              do i=1,im
                 tab(i,j)=cymx(i,j,n)
              enddo         
          enddo

         call pgadvance
         call pgvport(0.3,0.7,0.25,0.8)
         xi1=0.
         xi2=im-1
         yj1=0.
         yj2=jm-1
      call pgwindow(xi1,xi2,yj1,yj2)
      call pgsch(5.)
      call pgslw(2)
      call pgbox('bcints',100.,2,'bcints',100.,2)
      if(kyb.eq.1)call pglabel(' ','pixels ',' ')
      if(kyb.eq.2)call pglabel(' ','pixels ',' ')
c      call pgline(5,xdes,ydes)

c          write(3,*)'pggray: is,js,bbl,wwh:  ',is,js,bbl,wwh
         call PGGRAY(tab,1536,1536,1,im,1,jm,bbl,wwh,tr)
      enddo                     !  nm
c----------------------------      
      call pgend
      write(3,*)'234'
c       if(iflat2.eq.1)call system('okular flat2.ps &')
       write(3,*)' Sortie map3'
      endif                                !  iflat2
      return
      end
c=================================================
      subroutine solarobs(xr,yr,im,jm,nm,cal,sobsplot,sobstot)     
c                         in                 out
      dimension sobs(2000,200,24),cal(2000,200,24),sobsplot(2000,200,24)
C                              nmobs      
      dimension sobstot(2000,200,24,10),profnf(2000,200,250,3)
c                     i    j  n nmobs
      dimension xr(24,3,2),yr(24,3,2)
      
c      dimension xr(24,3,2),yr(24,3,2)
      character*38 fileb(10)
c     nfobs
      character*2880 chead
      integer*4 ima(1536,1536)   ! tabacer=sums    4 ?
      integer*2 tab2(1536,1536),imadark(1536,1536)
c      dimension des(200,2000)
      
c     Lecture
      call par1('    nfb1',nw,nfb1)
      call par1('    nfb2',nw,nfb2)
      call par1('  nfplot',nw,nfplot)
      write(3,*)' solarobs: nfb1,nfb2 ',nfb1,nfb2
      nfilesb=nfb2-nfb1+1
      call system('ls m*b1.fit > btab.lis')
      open(unit=11,status='old',file='btab.lis')
c      call system(' emacs btab.lis &')
      
      do 10 nf=nfb1,nfb2    !   all solar observations
         read(11,'(a)',iostat=ier,end=10) fileb(nf)
 10   continue
c 6    format(' solarobs: nf fileb(n!f) '(hi4,2x,a38)
c      write(3,6)' solarobs nf  fileb(nf) 'nf,fileb(nf)
c 10   continue

      call par1('  nfplot',1,nfplot)
      do 20 nf=nfb1,nfb2        
 7    format(' solarobs: nfplot, fileb(nfplot) ',i4,2x,a38)
      write(3,7)nf,fileb(nfplot)

      iu=12
      sundec=0.
      ipermu=1                  !  ms1
      iswap=1
      call par1('      is',nw,is)
      call par1('      js',nw,js)
      if(ipermu.eq.1)then
         isp=js    !  1024
         jsp=is    !  1536
      else
         isp=is
         jsp=js
      endif
      iswap=1
      call openold38(fileb(nf),sundec,iu)
        call counthead(iu,nhead,chead)
c                       in out   out
      if(nf.eq.nfplot)then   
                print *,' nhead= ',nhead
 4              format('head: ',a500)
         write(3,*)' solarobs  counthead:'
         write(3,4)chead
         print 4,chead
      endif
      
      ktab=1
      write(3,*)' readfits ipermu=',ipermu
      call readfits(iu,nhead,iswap,is,js,ipermu,tab2,
     1 ima,ktab)
c     !   permu dans readfits                       tabpermu
c-------------------------------------------------
c     subtract  ima - imadark
      imd=1024
      jmd=1536
      iux=31
      call mdark(iux,imadark,imd,jmd)       ! subtract dark
      do j=1,jmd
         do i=1,imd
            ima(i,j)=ima(i,j)-imadark(i,j)
         enddo
      enddo
c------------------------------
      imima=1024
      jmima=1536
      call  channels(xr,yr,ima,imima,jmima,sobs,iim,jjm,nm)   !  sobs
c                    in    in  1024  1536   out
      iic=(iim+1)/2
      jjc=(jjm+1)/2
c--------------------------------
      call calobs(cal,sobs,iim,jjm,nm) !    calib  sobs
c---------------------------------------- copie sobsplot
      if(nf.eq.nfplot)then
         do n=1,nm
            do j=1,jjm
               do i=1,iim
                  sobsplot(i,j,n)=sobs(i,j,n)
               enddo
            enddo
         enddo
         write(3,*)' solarobs nf  sobsplot(iic,jjc,nm) '
         write(3,*)(sobsplot(iic,jjc,n),n=1,nm)
 18   format(' solarobs cal(iic,61,n) ',9f6.2)
 19   format(' solarobs sobs(iic,61,n) ',9f6.0)
      write(3,18)(cal(iic,61,n),n=1,nm)
      write(3,19)(sobs(iic,61,n),n=1,nm)     
      endif      
c--------------------------------------copie sobstot
      do n=1,nm
        do j=1,jjm
           do i=1,iim
                  sobstot(i,j,n,nf)=sobs(i,j,n)
           enddo
        enddo
      enddo
 20   continue     !   nf
c--------------------------------------reprise de sobs
      if(nfplot.ne.0)then
       do n=1,nm
         do j=1,jjm
            do i=1,iim
               sobs(i,j,n)=sobstot(i,j,n,nfplot)     !  nf
            enddo
         enddo
      enddo
      endif
c************************************         
      write(3,*)' solarobs (fin) sobstot(iic,jjc,n,1)',
     1     (sobstot(iic,jjc,n,nfb1),n=1,nm) !   Bien
      write(3,*)' solarobs iim,jjm ',iim,jjm      
c      ivprof3       1    idem  ivprof3  fig 7 degree 3
      call par1('ivprof3',1,ivprof3)
      call par1('ivprof4',1,ivprof4)
      nfb1=1
      nfb2=2
      call ivmap3(sobstot,cal,iim,jjm,nm,nfb1,nfb2,profnf)
      call ivmap4(sobstot,cal,iim,jjm,nm,nfb1,nfb2,profnf)   
      return
      end
c===========================
      subroutine mdark(iux,imadark,imd,jmd)
      integer*2 imadark(1536,1536),kzd(512),lecx(1536)
      rewind(iux)
      read(iux)(kzd(n),n=1,512)
      write(3,*)' mdark  kzd  1-8 ',(kzd(n),n=1,8)
      print *,' mdark  kzd  1-8 ',(kzd(n),n=1,8)
      do j=1,jmd
         read(iux)(lecx(i),i=1,imd)
         if(j.eq.1.or.j.eq.jmd)then
            print *,' mdark j=1 lecx ',(lecx(i),i=1,imd,300)
            write(3,*)' mdark j=1 lecx ',(lecx(i),i=1,imd,300)
         endif
        do i=1,imd
           imadark(i,j)=lecx(i)
        enddo
      enddo
      write(3,*)' mdark imd,jmd, imadark(1,1) (1024,10536)',
     1                  imd,jmd, imadark(1,1),imadark(1024,1536)
      rewind(iux)
      return
      end
c=============================================
c*********************************************
      subroutine calobs(cal,sobs,im,jm,nm)
      dimension sobs(2000,200,24),cal(2000,200,24)
      do n=1,nm
         do j=1,jm  
            do i=1,im
               den=cal(i,j,n)
               sobs(i,j,n)=sobs(i,j,n)/den   !  correction cal
            enddo
         enddo
      enddo
      return
      end
c===================================      
