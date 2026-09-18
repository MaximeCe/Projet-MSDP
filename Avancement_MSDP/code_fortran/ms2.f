c ================================================================
c ms2.f - Calcul de la geometrie des canaux MSDP (Fortran 77)
c ================================================================
c Son role : determiner, pour chaque canal, la position precise de
c ses 4 coins dans l'image (flat - dark) au moyen de la detection
c des bords longs (i) et courts (j) via le gradient d'intensite.
c
c Sous-programmes :
c   - geom        : point d'entree appele par ms1.f
c   - SRECT       : preparation/detection (soustraction dark)
c   - SMAX        : interpolation parabolique du maximum
c   - newgeom     : detection moderne des bords + coins ABCDEF
c   - intersec    : intersection de deux droites (coins)
c   - plotgeo1/2/3 : trace des courbes de controle (PGPLOT)
c ================================================================
c
c ================================================
c geom : point d'entree, lit les parametres de
c geometrie puis appelle SRECT/newgeom.
c ================================================
      subroutine geom(nw,win,nm,iux,iuy,iuz,gname,istop,ima,ijcam,
     1     imima,jmima,xr,yr,imc,jmc)


      
       character*22 gname
       dimension kz(512)
       integer si,sgi,sj,sgj,distor,win(8)
       integer*2 ima(1536,1536) ! ,zmi(2000,3)
       dimension xr(24,3,2),yr(24,3,2)

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
       
      call par1('  interc',1,interc)
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
      call par1('    jeps',nw,jeps)
      call par1('   intvi',nw,intvi)
      call par1('   intvj',nw,intvj)
      call par1('    leps',nw,leps) 
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
      print *,' geom: i1,i2m,im,  j1,j2m,jm ',i1,i2m,im,  j1,j2m,jm
      write(3,*)' geom: i1,i2m,im,  j1,j2m,jm ',i1,i2m,im,  j1,j2m,jm
      write(3,5)
      print *,' call SRECT'
      call srect(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
     1           intvi,intvj,interc,
     2           si,sgi,sj,sgj,leps,
     3           nm,n1,n2,nc,n3n4,distor,
     4           iux,iuy,gname,
     5           ima,imima,jmima,
     6            milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
           write(3,5)

       return
       end    

c ================================================
c SRECT : soustrait le dark du flat, construit
c meanflat (flat-dark transpose) puis appelle newgeom.
c ================================================
      subroutine SRECT(milangi,milangj,i1,i2m,im,icp,j1,j2m,jm,lip,jeps,
     1                 intvi,intvj,intercan,
     2                 sip,sgip,sj,sgj,leps,nm,n1,n2,nc,n3n4,
     3                 distor,
     4                 iux,iuy,gname,
     5                 ima,imima,jmima,
     6                 milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr)
      integer*2 lec(1536),ima(1536,1536),lecx(1536),meanflat(4096,4096)
      dimension z(1536),zg(1536),x(1536),zmoy(1536),zmoyg(1536),
     1        xr(24,3,2),yr(24,3,2),yc(24,2,2),yca(24,2),
     2        nkbon(24,2),zmi(2000,3),y(1536),
     3        vi(24,6,6),vj(24,6,6),vmod(24,6,6),flech(24,2),ili(24,2),
     4        xdes(7),ydes(7),avi(6,6,3),avj(6,6,3),avmod(6,6,3),
     5       kz(512),zd(1536),xbord(24,5,2),ybord(24,5,2),
     6       integ(2,3),ecarti(24,6,6),ecartj(24,6,6),ecartmod(24,6,6),
     7       ksi(20),ksj(20),ksgi(20),ksgj(20)
      integer sip,sgip,si,sgi,sj,sgj,sgj1,sgj2,distor,displ,head(512)
      character titre*54,gname*22

      data ksi /2,2,4,4,4,6,6,6,9,9,9, 12,12,12,15,15,15,20,20,20/
      data ksgi/2,2,4,4,4,6,6,6,9,9,9, 12,12,12,15,15,15,20,20,20/
      data ksj /2,4,2,4,6,4,6,9,6,9,12, 9,12,15,12,15,20,15,20,25/
      data ksgj/2,4,2,4,6,4,6,9,6,9,12, 9,12,15,12,15,20,15,20,25/

      iug=34
      print 3,ksi
      
 3    format(20i3)
      call system('rm xryr.lis')
      open(unit=7,file='xryr.lis',status='new')
      
      displ=0


      nsm=20
      print *,' sip ',sip
        if(sip.ne.0)nsm=1
        ecmin=10000.
        nsol=nsm-1  ! default: last threshold (was 0 -> ksi(0) OOB)
        ipl=0
        ecartbis=1000.
      print *,' ecartbis ',ecartbis


c --- Lecture du flag dark et parametre d'angle de detection ---
      call par1('     idc',1,idc)
      call par1(' kdangle',1,kdangle)
         print *,' nsm ',nsm
c --- Boucle sur les seuils de detection (auto ou explicite) ---
      do 190 nseuils=1,nsm  !      ----------------------- nseuils
       if(nseuils.eq.nsm)ipl=1
1     format(' geom: nseuils, si,sgi,sj,sgj',i3,2x,4i4)

      if(sip.eq.0)then
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


        anglei=milangi*0.001
        anglej=milangj*0.001



      xa=0.1
      xb=0.6
       
      xc=0.65 !0.7
      xd=0.95 !0.9
      xcd=0.80  !0.5*(xc+xd)

        xcd1=0.75
        xcd2=0.85

        do2 n=1,24 !                             ????????????
        do kg=1,2
          do ihb=1,5
          xbord(n,ihb,kg)=0.
          ybord(n,ihb,kg)=0.
          enddo
          do ihb=1,3
          xr(n,ihb,kg)=0.
          yr(n,ihb,kg)=0.
          enddo
        enddo
2       continue

      i2=im-i2m
      idangle=float(kdangle)*(i2-i1)/1000.
          write(3,*)'idangle',idangle
      ic=0.5*(i1+i2)+0.5
      centre=0.5*(i1+i2)
      icentre=ic  

      j2=jm-j2m
         iap=ic-intvi
         ibp=ic+intvi
        do j=1,ijcam
        do ihmb=1,3
        zmi(j,ihmb)=0.
        enddo
        enddo

        print *,' read iux,iuy'
      rewind(iuy)
      read(iuy)kz

       rewind(iux)
       read(iux)kz
       print *,' dark head '
       print *,(kz(i),i=1,3)
       write(3,*)(kz(i),i=1,8)

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
         enddo
        if(j.eq.1)print *,' y-x 1,im,100 ',(lec(i),i=1,im,100)
        endif

        normsq=0
        do i=1,im
          ima(i,j)=lec(i)                              ! ima
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
 9    format(i8,2x,11i5)
         i1t=im*0.1
         i2t=im*0.9 
         write(3,*)' ima test   im,jm  i1t,i2t   ',im,jm,' ',i1t,i2t
         write (3,9)i1t,(ima(i1t,j),j=75,125,5)  
         write (3,9)i2t,(ima(i2t,j),j=75,125,5)    
 190  continue
 200  continue
       do j=1,im
          do i=1,jm
             meanflat(i,j)=ima(j,i)
          enddo
       enddo
c --- Lancement de la nouvelle geometrie sur meanflat (transpose) ---
       call newgeom(meanflat,jm,im)    !   New geometry
      
          return
          end
c ================================================
c SMAX : interpolation parabolique autour d'un
c maximum de gradient (affine la position du bord).
c ================================================
      subroutine SMAX(z,i,eps)
      dimension z(1536)
      b=z(i+1)-z(i-1)           !  /2 
      a=z(i+1)+z(i-1)-2.*z(i)   !  /2 
        if(a.eq.0.)return
      eps=-b/(2.*a)
     
      return
      end
c ================================================
c newgeom : noyau du calcul de geometrie. Detecte les
c bords sur 3 coupes, corrige la distorsion, calcule
c les coins ABCDEF. (array xx/yy en (20,9) - voir doc).
c ================================================
      subroutine newgeom(meanflat,im,jm)    !,xr,yr)
      integer*2 meanflat(4096,4096)
      dimension z(4096),zg(4096),zc(4096),zgc(4096),iedge(100,2),
     1  sig(2),ja(3),xx(20,40),yy(20,40),distort(2,40)
      print *,'newgeom (2164)'
      call par1('      nm',1,nm)
      call par1(' jtriple',1,jtriple)
      i1=5
      i2=im-4
      j1=1
      j2=jm
      call par1('     ja1',1,ja1)
      call par1('     ja2',1,ja2)
      call par1('     ja3',1,ja3)
      ja(1)=ja1
      ja(2)=ja2
      ja(3)=ja3
      write(3,*)' newgeom: ja 1,2,3 ',(ja(nn),nn=1,3)
      jc=ja(2)
      sig(1)=1.
      sig(2)=-1.
      call par1('    xdel',1,ixdel)
      xdel=float(ixdel)
      call par1(' mingrad',1,mingrad)
      grt=mingrad        
      zgt=grt
      call par1('  interp',1,interp)
 1    format(10i6)
      write(3,*)'  Newgeom   meanflat:'
      write(3,1)(meanflat(i,jc),i=1,im,15)
      write(3,*)' edges for 3 j-values'
      zmax=0.
      do i=1,im
         zc(i)=meanflat(i,jc)       !  jc
         zmax=amax1(zc(i),zmax)
      enddo

       zgmax=0.
      do i=1,im-1
        zgc(i)=zc(i+1)-zc(i)
        piv=abs(zgc(i))
        zgmax=amax1(zgmax,piv)
      enddo
      write(3,*)'  zmax,zgmax ',zmax,zgmax
      do i=1,im
         zc(i)=100.*zc(i)/zmax
         zgc(i)=100.*zgc(i)/zgmax
      enddo
      print *,'(2189)'
c --- Pour chacune des 3 coupes : bords gauche/droit par gradient ---
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
      enddo
      do i=i1,i2-1
         zg(i)=z(i+1)-z(i)
      enddo
      zg(i2)=zg(i2-1)
      print *,' zmax,zgmax ',zmax,zgmax     !  valeurs pour coupe centrale
      do i=i1,i2
            z(i)=100.*z(i)/zmax
            zg(i)=100.*zg(i)/zgmax
      enddo

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
        piv2=sig(is)*zg(i)
      if(piv2.lt.zgt)goto10
        piv1=sig(is)*zg(i-1)
        piv3=sig(is)*zg(i+1)
        if(piv2.lt.piv1.or.piv2.lt.piv3)goto10
        n=n+1
        if(n.gt.40)goto 10   !  garde-fou : borne tableau xx(20,40)/iedge(100,2)
        eps=0.5
        if(interp.eq.1)call smax(zg,i,eps)
 3      format(' edges: sig, l, n, iedge(n,is),zg(iedge-1/0/+1)',
     1     '  eps     XX      YY' )
 4    format(6x,f5.0,2i3,i6,5x,3f6.0,f6.2,2f8.2)
      iedge(n,is)=i
      xx(l,n)=iedge(n,is)+eps-1.
      yy(l,n)=ja(nj)-1.
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
      sig(1)=1.
      sig(2)=-1.
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
      do jj=jj1,jj2
               zmax=0.
               z(jj)=meanflat(ii,jj)
               zmax=amax1(zmax,z(jj))
      enddo
         zgmax=0.
      do jj=jj1,jj2-1
         zg(jj)=(z(jj+1)-z(jj))*sig(is)   !   sign
         piv=abs(zg(jj))
         zgmax=amax1(zgmax,zg(jj))
      enddo
      zg(jj2)=zg(jj2-1)
      print *,' l,n,zmax,zgmax ',l,n,zmax,zgmax
      do jj=jj1,jj2
            z(jj)=100.*z(jj)/zmax
            zg(jj)=100.*zg(jj)/zgmax
      enddo
      do 40 jj=jj1+1,jj2-1     
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
      write(3,*)' interp ',interp
c --- Calcul des coins A,B,C,D,E,F par intersection intersec ---
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
c ================================================
c intersec : intersection entre un bord long et un
c bord court (resout x=ay+b et y=cx+d).
c ================================================
      subroutine intersec(x1,x2,x3,x4,y1,y2,y3,y4,xres,yres)


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


c ================================================
c plotgeo1 : trace geo1.ps (profil, gradient, contours
c des canaux) ET ecrit le fichier ACDF2.lis.
c ================================================
      subroutine  plotgeo1(zc,zgc,zgt,i1,i2,im,jm,nm,xx,yy,ja)
      dimension zc(1536),zgc(1536),x(2),y(2),xdes(1536),ydes(1536)

      dimension xx(20,40),yy(20,40),ides(7),ja(3)
      data ides /11,12,13,16,15,14,11/



      
      print *,i1,i2,im,jm
      do i=1,im
         xdes(i)=i-1
      enddo
      im1=im-1
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
      call pgline(im,xdes,zc)                    !   intensity curve
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
      gradt=10
      grt=gradt+10.
      call pgsch(1.)
      call pgtext(120.,grt,'grt')
      grt=-gradt-15.
      call pgtext(80.,grt,'-grt')
      call pgsch(1.)

      call pgsls(1)
      x(2)=2
      
      call pgline(im,xdes,zgc)    !  gradient curve
      call pgvport(0.1,0.6,0.1,0.4)
      call pgvport(0.1,0.6,0.1,0.4)  
      call pgwindow(x1,x2,y1,y2)
      call pgbox('abcnts',200.,2,'bcnts',200.,2)

      write(3,*)'  ABCDEF X for nm channels'
 10   format(i4,2x,3f9.2,4x,3f9.2)
      do n=1,nm
         write(3,10)n,(xx(nd,n),nd=11,16)
      enddo
      write(3,*)'  ABCDEF Y for nm channels'
      do n=1,nm
         write(3,10)n,(yy(nd,n),nd=11,16)
      enddo
      call system('rm ACDF2.lis')
      open(unit=20,file='ACDF2.lis',status='new')
 91   format(8f8.2)
      do n=1,nm
      write(20,91)xx(11,n),xx(13,n),xx(14,n),xx(16,n),
     1            yy(11,n),yy(13,n),yy(14,n),yy(16,n)
      enddo
      close(20)
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
      return
      end

      
      




c ================================================
c plotgeo2 : trace geo2.ps (zoom premier canal, points
c abcdef/klmn/ABCDEF etiquetes).
c ================================================
      subroutine plotgeo2(xx,yy,xdel)
      dimension xx(20,40),yy(20,40),xdes(7),ydes(7)
      
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
      do nn=1,6
      call pgpt(6,xx(nn,1),yy(nn,1),8)
      enddo
      write(3,*)' geo2:'
      write(3,*)'xx / yy  for first channel'
      write(3,*)(xx(nn,1),nn=1,6)
      write(3,*)(yy(nn,1),nn=1,6)
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
      do klmn=7,16
         xdes(1)=xx(klmn,1)
         ydes(1)=yy(klmn,1)
         call pgpt(1,xdes(1),ydes(1),8)
      enddo
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
      

      call pgend
      call system('gv geo2.ps &')
      return
      end
  




c ================================================
c plotgeo3 : trace geo3.ps (variations de dimensions
c AC,DF,AD,CF en X et Y selon le canal).
c ================================================
      subroutine plotgeo3(xx,yy)    
      dimension xx(20,40),yy(20,40),xdes(40),ydes(40)
      call par1('      nm',1,nm)
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

      


