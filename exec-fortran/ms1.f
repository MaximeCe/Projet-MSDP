c ms1.f
c======
      character*38 file(4,200)
      character*2880 chead
      integer*4 tabpermu(1536,1536),tabaver(1536,1536)   ! tabacer=sums
      integer*2 tab2(1536,1536)
c               before permut ?
      integer*2 cymx(2000,200,24),ima(1536,1536),cliss(2000,200,24)
      integer sundec,uint,nfiles(4),nfa(4),nfb(4),head(512)
      character*22 xname,yname,zname,bname,name(4),xyname,gname
      character*7 cfile
c     names for dark, flat  and field stop
      integer*2 sort(1440),kswap1(1),kswap2(1)
      integer*4 t
      real*4 denom(2)
      dimension xr(24,3,2),yr(24,3,2),cal(2000,200,24)
c               geom                  calib
      integer win(2)            !   old code
c***********************************
c                  x              y              z              b
c     nxy=         1              2              3              4
c                dark            flat         field-stop       obs
c                list            list           list           list
c                read            read           read
c               average         average       average
c    ms2.f                   compute geom    comp geom
c                              subtract x    subtract x      
c    ms3.f                     channels       channels      
c                           compute cal
c                                                              read
c                                                             subtract x      
c                                                             channels (geom)
c                                                             calib   (cal)   
c****************************************************************************
c      ms1.f   
c
c
c
c     ms2.f    geom, srect             geometry
c                          CCD             coord imb,jmb
c                 (readfits)
c                 (xyplot)    y minus x    coord imb,jmb   flat1.ps
c                          ima             coord imc,jmc   xr,yr

c     ms3.f    channels  
c                 (coeff)
c                 (pix=      
c                 (maps3)               coord imd,jmd,nm    flat2.ps
c              calib                  calibration         cal.ps
c                 (plot_line)          
c                    (linecurv)       line shape in channels
c                    (trans)          wavelength translations, min. departures
c                 (profmean)          average profile

c                            Values of coordinates
c     CCD,       flat1.ps      imb,imima    1536     jmb,jmima   1024
c     XY permut.               imc          1024     jmc         1536
c   If  milsec=500 with
c     window                   li/1000       442"    lj/1000       61"
c   then      
c     channels   flat2.ps      imd           885     jmd          123 nm   9
c     filtergrams and doppler  ime         < 885     jme        < 123 nm   9    
c*******************************
     
       xname  ='x000000_00000000_00000'
       yname  ='y000000_00000000_00000'
       zname  ='z000000_00000000_00000'
       bname  ='b000000_00000000_00000'
       gname  ='g000000_00000000_00000'

c       call system('rm x170330_00000000_00000')
c       call system('rm y170330_00000000_00000')
c       open(unit=31,file='x170330_00000000_00000',
c     1     form='unformatted',status='new')
c       open(unit=32,file='y170330_00000000_00000',
c     1      form='unformatted',status='new')

      call system('rm channel.lis')
      open(unit=95,file='channel.lis',status='new')
       
      call system('rm ms.lis')
      open(unit=3,file='ms.lis',status='new')
      print *,'call readpar'
      call readpar
      
      sundec=0
      uint=0
      ipermu=1
      ijcam=1536

      call par1('    nfx1',nw,nfx1)
      call par1('    nfx2',nw,nfx2)
      call par1('    nfy1',nw,nfy1)
      call par1('    nfy2',nw,nfy2)
      call par1('    nfz1',nw,nfz1)
      call par1('    nfz2',nw,nfz2)
      call par1('    nfb1',nw,nfb1)
      call par1('    nfb2',nw,nfb2)
      nfa(1)=nfx1
      nfb(1)=nfx2
      nfiles(1)=nfx2-nfx1+1
      nfa(2)=nfy1
      nfb(2)=nfy2
      nfiles(2)=nfy2-nfy1+1
      nfa(3)=nfz1
      nfb(3)=nfz2
      nfiles(3)=nfz2-nfz1+1
      nfa(4)=nfb1
      nfb(4)=nfb2
      nfiles(4)=nfb2-nfb1+1

      call par1('      is',nw,is)
      call par1('      js',nw,js)
      if(ipermu.eq.1)then
         isp=js
         jsp=is
      else
         isp=is
         jsp=js
      endif
      iswap=1
      
c      nfmax=50
 1    format(38a)
 5    format(20i5)
c----------------------------------------------loop xy      dark/flat
      do 300 nxy=1,2            !4
               write(3,*)' '
      write(3,*)' Loop nxy ',nxy
         print *,' '
         print *,' nxy,nfa,nfb ',nxy,nfa(nxy),nfb(nxy)
         
c         if(nfa(nxy)*nfb(nxy).ne.0)then
      print *,' '
      if(nxy.eq.1)then
         print *,' DARK'
         call system('ls m*x1.fit > xtab.lis')
         print 1,' xtab.lis'
         open(unit=11,status='old',file='xtab.lis')
c     call system('emacs xtab.lis &')
c         rewind(11)
      endif
      if(nxy.eq.2)then
         print *,' FLAT'
         call system('ls m*y1.fit > ytab.lis')    !     y -> b
         print 1,' ytab.lis'
         open(unit=12,status='old',file='ytab.lis')
c         call system('emacs ytab.lis &')
c        rewind(12)
      endif
c      if(nxy.eq.4)then
c         print *,' OBS'
c         call system('ls m*b1.fit > btab.lis')    !     y -> b
c         print 1,' btab.lis'
c         open(unit=14,status='old',file='btab.lis')
c         call system('emacs btab.lis &')
c      endif
c      if(nxy.gt.2)goto 300
c                                                 loop files X or y
      
      do 100 nf=1,nfb(nxy)
      if(nxy.eq.1)read(11,'(a)',iostat=ier,end=100) file(nxy,nf)
      if(nxy.eq.2)read(12,'(a)',iostat=ier,end=100) file(nxy,nf)
 6    format(' nf file(nxy,nf) ',i4,2x,a38)
      write(3,6)nf,file(nxy,nf)
      
c      if(ier.ne.0) goto 3
      
      if(nxy.eq.1)then
         if(nf.eq.nfb(1))then
      xname(1:1)=file(nxy,nf)(33:33)   !   name of average dark
      xname(2:17)=file(nxy,nf)(17:32)
      print *,' xname = ',xname
      write(3,*)' xname ',xname
      name(1)=xname
         endif
      endif
      if(nxy.eq.2)then
         if(nf.eq.nfb(2))then
      yname(1:1)=file(nxy,nf)(33:33)   !   name of average flat
      yname(2:17)=file(nxy,nf)(17:32)
      zname(1:1)=file(nxy,nf)(33:33)   !   name of average field stop
      zname(2:17)=file(nxy,nf)(17:32)
      print *,' yname= ',yname
      write(3,*)' yname ',yname
      name(2)=yname
      name(3)=zname  !   name of average field stop
      gname(2:17)=file(nxy,nf)(17:32)
      name(4)=gname
          endif
      endif
c                        *****
c     nfiles(nxy)=nfiles(nxy)+1       !  total  number of files
 3    continue
 100  continue      ! nfb(nxy)
c 3    continue
      write(3,*)' nxy,  useful nfiles(nxy)= ',nxy,nfiles(nxy)

         if(nxy.eq.1)rewind(11)
         if(nxy.eq.2)rewind(12)
         
c     Average  nxy   --------------------
         write(3,*)' '
         write(3,*)' Average nxy ',nxy
         write(3,*)' from file ',nfa(nxy),' to ',nfb(nxy)
      do j=1,jsp
         do i=1,isp
            tabaver(i,j)=0
         enddo
      enddo
c-----------------------------------read files
      iu=20+nxy
      do 200 nf=nfa(nxy),nfb(nxy) !1,nfiles(nxy)
         write(3,*)' nxy,nf,file(nxy,nf) ', nxy,nf,file(nxy,nf) 
         ktab=0
         if(nf.eq.nfa(nxy))ktab=1
 4       format('head: ',a500)
c----------------------------------------------------------------      
c      if(nxy.eq.2) then                              !   nxy=2
            print *,' file(nf) ',file(nxy,nf)
         call openold38(file(nxy,nf),sundec,iu)
         call counthead(iu,nhead,chead)
c                       in out   out
      
      if(nf.eq.nfa(nxy))then   
      print *,' nhead= ',nhead
         write(3,4)chead
         print 4,chead
      endif
           write(3,*)' readfits ipermu=',ipermu
      call readfits(iu,nhead,iswap,is,js,ipermu,tab2,tabpermu,ktab)
c     !   permu dans readfits
      write(3,*)' nxy=',nxy,'  file(',nf,') '
c-------------------------
      if(nf.ge.nfa(nxy))then
      do ip=1,isp
      do jp=1,jsp
         tabaver(ip,jp)=tabaver(ip,jp)+tabpermu(ip,jp)
      enddo
      enddo
      endif

 8    format(' nf=',i3,'  ABCD before permut ',4i5,' BCDA after ',4i5)
      print 8,nf,tab2(1,1),tab2(is,1),tab2(is,js),tab2(1,js),
     1 tabpermu(isp,1),tabpermu(isp,jsp),tabpermu(1,jsp),tabpermu(1,1)

 200  continue                  ! end loop nf
c-----------------------------------averages
      print *,' AVERAGE'
      if(nxy.eq.1)print *,'xname  ',xname
      if(nxy.eq.2)print *,'yname  ',yname
c     piv=float(nfiles(nxy))
      print *,' isp,jsp ',isp,jsp            !   avant erreur ?

      kpiv=nfb(nxy)-nfa(nxy)
      denom(nxy)=float(kpiv)+1.
      write(3,*)' nxy, denom(nxy) ',nxy,denom(nxy)
      do jp=1,jsp
      do ip=1,isp
         tab2(ip,jp)= float(tabaver(ip,jp))/denom(nxy) +0.5
      enddo
      enddo
c-----------------------------------------------------  open x,y files
 7    format(15i5)
 9    format(i4,2x,15I5)
 11   format(' ',a22,'    denom=',f6.3,'  extreme points  ', 4i5)
c-----------------------------------average array
      head(1)=3
      head(2)=isp    !1024
      head(3)=jsp    !1536
      head(4)=1
c        isp=1024
c        jsp=1536
      do n=4,512
         head(n)=0
      enddo

      print *,' write files xname,yname '
      iut=30+nxy
         print *, name(iut)
         call system('rm '//name(nxy))
       open(unit=iut,file=name(nxy),
     1      form='unformatted',status='new')
c       open(unit=33,file=name(3),
c     1      form='unformatted',status='new')
       write(3,*)' write iut,head(2),head(3),isp,jsp',
     1                   iut,head(2),head(3),isp,jsp                    
       write(iut)(head(n),n=1,512) !   xname
         do j=1,jsp
         write(iut)(tab2(i,j),i=1,isp)
      enddo
 290  format(i8,2x,11i5)
      
         i1t=isp*0.1
         i2t=isp*0.9
         write(3,*)' tabpermu test   nxy=',nxy
         write (3,290)i1t,(tab2(i1t,j),j=75,125,5)
         write (3,290)i2t,(tab2(i2t,j),j=75,125,5)
  
c------------      
      write(3,11)name(nxy),denom(nxy),
     1        tab2(1,1),tab2(isp,1),tab2(isp,jsp),tab2(1,jsp)
      
      do j=1,jsp,100
      write(3,9)j,(tab2(i,j),i=1,isp,100)
      enddo
c---------------------------------------------------------------      
c         endif                                  !   nfa*nfb ne 0
 300  continue                 ! end loop nxy
      close(11)
      close(12)
      close(14)
      close(21)
      close(22)
      rewind(31)
      rewind(32)
      rewind(33)      
c===================================================   geometry   ms2.f
      nw=1
      win(1)=1
      win(2)=0
      nm=9
c     CCD,ima=obs      imb    1536     jmb   1024
c     permut,maps      imc    1024     jmc   1536
c     channels, c*     imd     885     jmd    123     nm   9
      

      imb=1536
      jmb=1024
      imc=1024
      jmc=1536
c      imd=885
c      jmd=123
      nm=9
c------------------------------------------
      write(3,*)' enter geom'
      call geom(nw,win,nm,31,32,32,gname,istop,ima,ijcam,imima,jmima,
     1                                              xr,yr,imc,jmc)
c     ---                  x  y     g          OUT
      write(3,*)' end geom'
c============================================= Calibration    ms3.f
c      write(3,*)' entree channels'
c      call channels(xr,yr, ima, imc,jmc, cymx,imd,jmd,nm,1)
c                   in     IN            OUT             in
c     y minus x format c  kyb(plots y or b)
c      write(3,*)' end channels '
c----------------------------------cal.ps
c      print *,' call calib: imd,jmd,nm ',imd,jmd,nm
c      call calib(cymx,imd,jmd,nm,cliss,jtr,cal,xr,yr)
c                IN      
c      write(3,*)' calib '
c      call pgend
c---------------------------------          files m*-> c*  d*

      
c      close(unit=31)
c      close(unit=32)
c      close(unit=33)
      close(unit=95)
      close(unit=3)
           
      stop
      end
c***********************************************
      subroutine readpar
      character*8 nom
      open(unit=96,file='ms.par',status='old')
      rewind(96)

      do n=1,1000         ! 1000
 1       format(a8,i8)
         read(96,1)nom,nombre
      write(3,*)n,nom,nombre
      print *,n,nom,nombre
        if(nom.eq.'end     ')goto 2
      enddo
   
 2    close(unit=96)
      return
      end
c------------------------------------------
      subroutine par1(name,nw,nombre)
      character*8 nom,name
      open(unit=96,file='ms.par',status='old')
      rewind(96)

      do n=1,1000              
 1       format(a8,i8)
 2       format(' par1 ',a8,2i8)
            read(96,1)nom,nombre
            if(nom.eq.name)then
               write(3,2)nom,nw,nombre
               print 2,nom,nw,nombre
               return
            endif
        if(nom.eq.'end     ')goto 3
      enddo
 4    format(' par1 ',a8,'  no  ')
 3    print 4,name
      close(unit=96)
      return
      end
c*******************************************
      Subroutine readfits
     1     (iu,inbhead,iswap,is,js,ipermu,tab2,tabpermu,ktab)
c                                             permuted
       integer*2 tab2(1536,1536)
       integer*4 tabpermu(1536,1536)
       integer*2 sort(1440),in(1),out(1),ku
       integer t,uint
       write(3,*)' readfits iu,inbhead,is,js,ipermu,ktab',
     1                     iu,inbhead,is,js,ipermu,ktab   
c       call par1('    uint',1,uint)       
c                            
c Lecture de tab(is,js)
        n=inbhead
        i=0
        j=1
        k=1
 100    n=n+1
c      if(n.eq.10.or.n.eq.300)
c     1  print *,'n,sort ',n,(sort(t),t=1,1440,200) ! ****
        read(iu,rec=n,iostat=ios)  (sort(t),t=1,1440)
c        if(n.eq.2)  print *,' sort(t),t=1,10 ',(sort(t),t=1,10)
c     print *,'apres read n =',n
c 101    format(' n,(sort(t),t=1,1440,100) ',11i6)
c        if(n.lt.3)print 101,n,(sort(k),k=1,1440,100)
        
        if(ios.lt.0) go to 1000
        t=1
 200    i=i+1

        if(i.le.is) go to 500
         if(j.lt.js) then

           j=j+1               !   loop j
           i=1
         else
          goto1000
c            if(k.lt.nclim) then
c               k=k+1
c               j=1
c               i=1
c            else
c               go to 1000
c            endif
          endif

 500      continue
            lswap=0
            if(i.eq.is/2.and.j.eq.js/2)lswap=1
          if(iswap.eq.1)then
              in(1)=sort(t)
              call swap(in,1,out,lswap)
              sort(t)=out(1)
          endif
c          if(uint.eq.1)then
c             ku=sort(t)
c             call sint(ku)
c             sort(t)=ku
c          endif
          tab2(i,j)=sort(t)     ! before prmut

       t=t+1
       if (t.eq.1441) go to 100
       go to 200                   !   loop i
1000   continue
c--------------------
       if(ipermu.eq.1)then
          ips=js
          jps=is
         do i=1,is                  !  1536
            jp=i     
            do j=1,js               !  1024
               ip=js+1-j    
               tabpermu(ip,jp)=tab2(i,j)
            enddo
         enddo
      else
         ips=is
         jps=js
        do j=1,js                  !  1536
           do i=1,is
            tabpermu(i,j)=tab2(i ,j)
           enddo
         enddo
       endif

       if(ktab.eq.1)then
c     print *,'lecture des donnees en I2 effectuee'
       print *,' tab2 before permut'
c       isp=js
 4     format(i5,16i5)
       do i=1,is,100    !50                   
          print 4,i,(tab2(i,j),j=1,js,100)  ! 100
       enddo

       print *,' after' 
       do jp=1,jps,100     !1041,1040                  !  ?????????
          print 4,jp,(tabpermu(ip,jp),ip=1,ips,100)
       enddo
      endif
          return
          end
c----------------------------------
c      subroutine sint(k)
c                ----
c      integer*2 k

c      if(k.ge.0)then
c        kpiv=k
c      else
c        kpiv1=k
c        kpiv=kpiv1+65536
c      endif

c        k=kpiv/2
c        return
c      end
c----------------------------------                 
       subroutine swap(in,ncar,out,lswap)   
c                 ----
c       in = entree
c       out= sortie
c       ncar=nombre de integer*2 a swaper
c
      integer*2 in(1),out(1),auxi2
      logical*1 low,auxl1(2)
      equivalence (auxi2,auxl1(1))
c
      do i=1,ncar
      auxi2=in(i)
      low=auxl1(1)
      auxl1(1)=auxl1(2)
      auxl1(2)=low
      out(i)=auxi2
      enddo
      if(lswap.eq.1)write(3,*)' lswap in auxl1(1) auxl1(2) out ',
     1              in(1), auxl1(1),auxl1(2),out(1)
      return
      end
c----------------------------------
      subroutine counthead(iu,nb,chead)
c                          in out      
      integer nb,head(512)
      character chead*2880
               write(3,*)' subroutine counthead'
      do nb=1,10
      read(iu,rec=nb) chead(1:2880)
c         head=buf(1:500)

      i=1
         do n=1,36 
         if(chead(i:i+3).eq.'END ') go to 1000
         i=i+80
         enddo
      enddo
 1000 continue
      return
      end
c----------------------------------      
      subroutine openold38(name1,sundec,iu)
      character*38 name1
      integer sundec
      character*38 cfile
      if(sundec.eq.1) then
      open(unit=iu,status='old',form='unformatted',recl=720,
     1access='direct',file=name1)
      else
      open(unit=iu,status='old',form='unformatted',recl=2880,
     1access='direct',file=name1)           
      endif
      return
      end
c----------------------------------------------
      subroutine opennew22(xyname,iu)
      character*22 xyname
      open(unit=iu,file=xyname,status='new')
      return
      end
c-----------------------------------------------   
      subroutine openold22(name,sundec,iu)
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='old',form='unformatted',file=name)
        write(3,*)'ouvert'
      return
      end
c------------------------------------
      subroutine opennew22sf(name,sundec,iu)
c                                           sf = avec format
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='new',form='formatted',file=name)
c        write(3,*)'ouvert'
      return
      end
c------------------------------------
            subroutine comptehead(iu,nb)     !  double counthead????
c                           in out
      integer nb
      character buf*2880
      character cc*600
         write(3,*)' subroutine comptehead iu=',iu
      do nb=1,10
         read(iu,rec=nb) buf(1:2880)
         cc=buf(1:600)
 1       format('buf(1:600): ',a600)
         write(3,1)cc
      i=1
      do n=1,36 
      if(buf(i:i+3).eq.'END ') go to 1000
      i=i+80
      enddo
      enddo
1000  return
      end
c-----------------------------------------------
      subroutine openold22sf(name,sundec,iu)
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='old',form='formatted',file=name)
c        write(3,*)'ouvert'
      return
      end
c---------------------------------------
      subroutine DPMCAR(X,Y,P,ND,NT,COEF)
c                ------
c Ce sous-programme [ Schneider 75 ] calcule par une 
c methode de moindres carres en double precision le 
c polynome de NT termes (10 max) Y=F(X) associe a ND mesures.
C
C P=poids
c
      DOUBLE PRECISION C,F,PIVOT
      DIMENSION X(2000),Y(2000),P(2000),COEF(10),F(10),C(11,11)
      IF(NT.GT.10)NT=10
      NTP=NT+1
C
      DO 2 I=1,NT
      DO 2 L=1,NTP
2     C(L,I)=0.
C
      DO10 L=1,ND
      F(1)=1
        DO7 I=2,NT
C                   CHANGEMENT D'ECHELLE X
7       F(I)=F(I-1)*X(L)/1000.
      DO10 M=1,NT
        DO9 K=M,NT
9       C(K,M)=C(K,M)+F(K)*F(M)*P(L)
10    C(NTP,M)=C(NTP,M)+F(M)*P(L)*Y(L)
C
      DO20 M=1,NTP
      DO20 K=M,NTP
20    C(M,K)=C(K,M)
C
      DO104 I=1,NT
      PIVOT=C(I,I)
        DO103 M=1,NT
103     C(M+1,I)=C(M+1,I)/PIVOT
      DO104 K=1,NT
      IF(I.EQ.K)GOTO104
      PIVOT=C(I,K)
        DO105 M=1,NT
105     C(M+1,K)=C(M+1,K)-C(M+1,I)*PIVOT
104   CONTINUE
C
      COEF(1)=C(NTP,1)
      DO106 I=2,NT
106   COEF(I)=C(NTP,I)/(1000.**(I-1))
      RETURN
      END
c------------------------------
