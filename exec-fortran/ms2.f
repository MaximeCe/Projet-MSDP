c ms2.f
c========
c                                        bmg1.f, bmc1.f, cmf1.f
c**************************
c  Programme geom                        bmg1 - UNIX
c*************************************************************************
      subroutine geom(nw,win,nm,iux,iuy,iuz,gname,istop,ima,ijcam,
c                                                       out
     1     imima,jmima,xr,yr,imc,jmc)
c     out............

c     CCD        imb    1536     jmb   1024
c     permut     imc    1024     jmc   1536
c     channels   imd     885     jmd    123     nm   9

      
c       character comm*80,fin*3,end*3
       character*22 gname
       dimension kz(512)
       integer si,sgi,sj,sgj,distor,win(8)
c       integer winp,sgj1,sgj2
       integer*2 ima(1536,1536) ! ,zmi(2000,3)
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
      write(3,5)
      print *,' call SRECT'
      call srect(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
c          -----
     1           intvi,intvj,interc,
     2           si,sgi,sj,sgj,leps,
     3           nm,n1,n2,nc,n3n4,distor,
     4           iux,iuy,gname,
     5           ima,imima,jmima,
     6            milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
           write(3,5)

c        if(istop.eq.1)stop
       return
       end    

c-------------------------------------------------------------------
      subroutine SRECT(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
     1                 intvi,intvj,intercan,
     2                 sip,sgip,sj,sgj,leps,nm,n1,n2,nc,n3n4,
     3                 distor,
     4                 iux,iuy,gname,
     5                 ima,imima,jmima,
     6                 milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
      integer*2 lec(1536),ima(1536,1536),lecx(1536),meanflat(1536,1024)
c                                                 ------------------
c                                                  computed in xyplot      
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
      do 190 nseuils=1,nsm  !      ----------------------- nseuils
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
       write(3,*)(kz(i),i=1,8)
c       im=kz(2)
c       jm=kz(3)

      iia=i1
      iib=i1+4
      iic=i2-4
      iid=i2

      print *,' subtraction y-x    im,jm ',im,jm
      imima=im
      jmima=jm
      do j=1,jm
         read(iuy)(lec(i),i=1,im)
         if(j.eq.1)print *,' lec 1,im,100 ',(lec(i),i=1,im,100)
         
        if(idc.eq.1)then                               ! oui idc=1
        read(iux)(lecx(i),i=1,im)
         do i=1,im
           lec(i)=lec(i)-lecx(i)
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
         write (3,9)i1t,(ima(i1t,j),j=75,125,5)  
         write (3,9)i2t,(ima(i2t,j),j=75,125,5)    
c============================================
 190  continue
 200  continue
c                calcul de imaperm puis meanflatmd   (meanflat minus dark)
      do j=1,1024
         do i=1,1546
            meanflat(i,j)=ima(j,i)
         enddo
      enddo
c-------------------------
      call newgeom(meanflat)    !   New geometry
      
         return
         end
c=============================================
c--------------------------
      subroutine SMAX(z,i,eps)
      dimension z(1536)
c      eps=0.5
c      call par1('  interp',1,interp)
c     parabolic interpolation of intensity gradients
c      write(3,*)' smax: interp eps',interp,eps
c      if(interp.eq.1)then 
      b=z(i+1)-z(i-1)           !  /2 
      a=z(i+1)+z(i-1)-2.*z(i)   !  /2 
        if(a.eq.0.)return
      eps=-b/(2.*a)
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
      subroutine newgeom(meanflat)    !,xr,yr)
      integer*2 meanflat(1536,1024)
      dimension z(1536),zg(1536),zc(1536),zgc(1536),iedge(100,2),
     1  sig(2),ja(3),xx(20,9),yy(20,9),distort(2,9)
c     ,xr(24,3,2),yr(24,3,2)
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
      jtriple=1        !       j-1,j,j+1
      i1=5
      i2=im-4
      j1=1
      j2=jm
      ja(1)=1+150
      ja(2)=1+500
      ja(3)=1+850
c             -----      
c      ja(2)=(ja(1)+ja(3))/2     ! jc= 512
      write(3,*)' newgeom: ja 1,2,3 ',(ja(nn),nn=1,3)
c     jc=(jm+1)/2   !  512,
      jc=ja(2)
      sig(1)=1.
      sig(2)=-1.
      call par1(' mingrad',1,mingrad)
c                           minimum threshold of intensity gradient
      grt=mingrad        
      zgt=grt   !  minimum threshold of gradient for the i of maximum intensity
      call par1('  interp',1,interp)  !  parabolic interpolation with gradients
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
      print *,'(2189)'
      do30 nj=1,3               !  j de la coupe
      print *,'nj',nj
      jj=ja(nj)
      print *,'jj ',jj
      write(3,*)' '
      write(3,*)' ja(',nj,')= ',jj

      do i=i1,i2
         z(i)=meanflat(i,jj)
         if(jtriple.eq.1)
     1     z(i)=(meanflat(i,jj-1)+meanflat(i,jj)+meanflat(i,jj+1))/3.
c         zmax=amax1(zmax,z(i))
      enddo
c      zgmax=0.
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

c      call plotgeo1(z,zg,zgt,i1,i2,im,jm)
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
         
         do20 is=1,2                     !   signe du gradient
         if(is.eq.1)then
            l=nj
         else
            l=nj+3
         endif
      write(3,3)
             n=0
      do 10 i=i1+1,i2-1
c            if(z(i).lt.zseuil)goto10
        piv2=sig(is)*zg(i)
      if(piv2.lt.zgt)goto10
        piv1=sig(is)*zg(i-1)
        piv3=sig(is)*zg(i+1)
        if(piv2.lt.piv1.or.piv2.lt.piv3)goto10
        n=n+1
        eps=0.5
        if(interp.eq.1)call smax(zg,i,eps)
 3      format(' edges: sig, l, n, iedge(n,is),zg(iedge-1/0/+1)',
     1     '  eps     XX      YY' )
 4    format(6xf5.0,2i3,i6,5x,3f6.0,f6.2,2f8.2)
      iedge(n,is)=i
      xx(l,n)=iedge(n,is)+eps-1.
      yy(l,n)=ja(nj)-1.
c                                    
      write(3,4)sig(is),l,n,iedge(n,is),zg(i-1),zg(i),zg(i+1),eps,
     1         xx(l,n),yy(l,n)
 10   continue
 20   continue                  !  left,right
 30   enddo                     !  nj              abcdef  canaux

      do n=1,nm
         xx(15,n)=xx(5,n)   ! E
         yy(15,n)=yy(5,n)
         xx(12,n)=xx(2,n)   ! B
         yy(12,n)=yy(2,n)
      enddo
c----------------------------      
c     distortion:  flèches entre abc   et    def
      valqm=0.
      do n=1,nm
         distort(1,n)=xx(2,n)-(xx(1,n)+xx(3,n))/2.
         distort(2,n)=xx(5,n)-(xx(4,n)+xx(6,n))/2.
         valqm=valqm+distort(1,n)**2+distort(2,n)**2
      enddo                     !    nm=9
      valqm=valqm/(2.*float(nm))
      valqm=sqrt(valqm)
 31   format(' distortion: ',f6.3,
     1      '  quadratic mean value in pixel-to-pixel distance ')
      write(3,31)valqm
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
c               zmax=0.
               z(jj)=meanflat(ii,jj)
c               zmax=amax1(zmax,z(jj))
      enddo
c         zgmax=0.
      do jj=jj1,jj2-1
         zg(jj)=(z(jj+1)-z(jj))*sig(is)   !   sign
c         piv=abs(zg(jj))
c         zgmax=amax1(zgmax,zg(jj))
      enddo
      zg(jj2)=zg(jj2-1)
      print *,' l,n,zmax,zgmax ',l,n,zmax,zgmax
      do jj=jj1,jj2
            z(jj)=100.*z(jj)/zmax
            zg(jj)=100.*zg(jj)/zgmax
      enddo
c                                            zgmax           
      do 40 jj=jj1+1,jj2-1     
c            if(z(i).lt.zseuil)goto10  inutile 
        piv2=zg(jj)
      if(piv2.lt.zgt)goto 40                 !  zgt threshold
        piv1=zg(jj-1)
        piv3=zg(jj+1)
        if(piv2.lt.piv1.or.piv2.lt.piv3)goto40
        eps=0.5
        if(interp.eq.1)call smax(zg,jj,eps)
      xx(l,n)=ii-1
      yy(l,n)=jj-1+eps
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

      call plotgeo1(zc,zgc,grt,i1,i2,im,jm,nm,xx,yy,ja)
      call plotgeo2(xx,yy,xdel)   !  ,xr,yr)

      call plotgeo3(xx,yy)
      
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
      subroutine  plotgeo1(zc,zgc,zgt,i1,i2,im,jm,nm,xx,yy,ja)
      dimension zc(1536),zgc(1536),x(2),y(2),xdes(1536),ydes(1536)

      dimension xx(20,9),yy(20,9),ides(7),ja(3)
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
      enddo
      im1=im-1
c-----------------------------------------------  plot geo1.ps
      call pgbegin(0,'geo1.ps/ps',1,1)
      call pgvport(0.1,0.6,0.7,0.9)  !                       (0.1,0.9,0.5,0.9)
         call pgslw(3)
         call pgsch(1.)
         call pgsls(1)      
                 x1=0.
                 x2=1536.-1.
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

      call pgsls(2)
      x(1)=0
      x(2)=im-1
      y(1)=zgt                  !  +
      y(2)=y(1)
      call pgline(2,x,y)
      y(1)=-y(1)               !  - 
      y(2)=y(1)
      call pgline(2,x,y)
c                                  thresholds for gradient
      gradt=10
      grt=gradt+10.
c              sgi2=sgi+10
c        sgi3=-sgi-25
      call pgsch(1.)
      call pgtext(120.,grt,'grt')
      grt=-gradt-15.
      call pgtext(80.,grt,'-grt')
      call pgsch(1.)

      call pgsls(1)
      x(2)=2
      
      call pgline(im,xdes,zgc)    !  gradient curve
      call pgvport(0.1,0.6,0.1,0.4)
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
      call system('gv geo1.ps &')
c--------------------------------------------    
      return
      end
c==========================================
c      call pgvport(xa,xb,.50,.70)
c      call pgwindow(x1,x2,-100.,100.)
c      call pgbox('abcnts',200.,2,'bcnts',50.,5)

c     if(ipl.ne.0)call pgwindow(x1,x2,y1,y2)
c      if(ipl.ne.0)call pgbox('bcnts',200.,2,'bcnts',200.,2)
c      if(ipl.ne.0)call pglabel('J','I',' ')
      
c      if(ipl.ne.0)call pgvport(xc,xcd1,0.5,0.9)
c      if(ipl.ne.0)call pglabel(' ',' ','I')
c      if(ipl.ne.0)call pgvport(xcd1,xcd2,0.5,0.9)
c      if(ipl.ne.0)call pglabel(' ',' ','J')
c      if(ipl.ne.0)call pgvport(xcd2,xd,0.5,0.9)
c      if(ipl.ne.0)call pglabel(' ',' ','modulus')
c
      
c       indices      xbord (X//i)
c                       ybord (y//j)
c                  (canal n)
c                 
c                                  n,5,1   n,5,2
c                                  C-l-------n-F          13  8 10  16
c                                  | |       | |   
c                           n,3,1  c..       ..f  n,3,2     3       6
c                                  |           |
c                           n,2,1  b           e  n,2,2    12       15
c                                  |           |
c                           n,1,1  a           d  n,1,2     1       4
c                                  |           |
c                                  A-k-------m-D           11 7   9 14
c   
c                                  n,4,1   n,4,2



c     xr(n,l,k1,k2) l=letter, k1=bottom/top, k2=meft/right
c     yr(n,l,k1,k2)
c
c                  (n,4,2,1)    (n,3,2,1)  (n,3,2,2)  (n,4,2,2)
c                       Dtl          C2        Ctr        Dtr
c                        \          |          |         /
c                         +---------+----------+--------+
c                         |                             |
c      jc+jd    (n,2,2,1) +                             +
c                   B2   |                              |     B4
c                         |                             |
c                         |                             |
c      jc       (n,1,1,1) +                             +
c                   A1          |                             A2
c                         |                             |
c                         |                             |
c      jc-jd    (n,2,1,1) +                             +     B3
c                   B1          |                             |
c                         +---------+----------+--------+
c                        /          |          |         \
c                  D(n,4,1,1)  C(n,3,1,1)     C(n,3,1,2)   D(n,4,1,2)
c
c                                icn-id      icn+id     
c
c       
c

c======================================
      subroutine plotgeo2(xx,yy,xdel)
c                               +/-decalages en X pour klmn      
c     dimension  xr(24,3,2),yr(24,3,2),xdes(7),ydes(7)
c      dimension  xr(24,3,2),yr(24,3,2)
      dimension xx(20,9),yy(20,9),xdes(7),ydes(7)
c      xdel=25   !shift for k,l,m,n
      
      n=1

      call pgbegin(0,'geo2.ps/ps',1,1)
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
      call system('gv geo2.ps &')
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
      subroutine plotgeo3(xx,yy)    
      dimension xx(20,9),yy(20,9),xdes(9),ydes(9)
      nm=9
      nc=5
      xnm1=nm+1.
      write(3,*)' plotgeo3: '
      call par1('  interp',1,interp)
      
      if(interp.eq.1)call pgbegin(0,'geo3.ps/ps',1,1)
      if(interp.eq.0)call pgbegin(0,'geo3b.ps/ps',1,1)
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
      call system('gv geo3.ps &')
      return
      end
c=============================================

      
c     call pgvport(0.1,0.9,.1,.5)
c      call pglabel('X (unit=arcsec/2) ','Intensity gradient',' ')
c      call pgwindow(x1,x2,-100.,100.)
c      call pgbox('abcnts',200.,2,'bcnts',50.,5)

c--------               abcdef    klmn     ABCDEF
c                       1    6    7  10   11    16

