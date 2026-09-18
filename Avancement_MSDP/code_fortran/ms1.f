c ================================================================
c ms1.f - Programme principal du pipeline MSDP (Fortran 77)
c ================================================================
c 1. Lit les parametres du fichier ms.par
c 2. Calcule les fichiers moyens dark (x) et flat (y)
c 3. Appelle le calcul de geometrie des canaux (ms2.f)
c
c Les sous-programmes utilitaires (lecture FITS brute, swap
c d'octets, moindres carres, ouverture de fichiers) sont
c regroupes a la fin de ce fichier.
c ================================================================
c
      character*38 file(4,200)
      character*2880 chead
      integer*4 tabpermu(1536,1536),tabaver(1536,1536)   ! tabacer=sums
      integer*2 tab2(1536,1536)
      integer*2 cymx(2000,200,24),ima(1536,1536),cliss(2000,200,24)
      integer sundec,uint,nfiles(4),nfa(4),nfb(4),head(512)
      character*22 xname,yname,zname,bname,name(4),xyname,gname
      character*7 cfile
      integer*2 sort(1440),kswap1(1),kswap2(1)
      integer*4 t
      real*4 denom(2)
      dimension xr(24,3,2),yr(24,3,2),cal(2000,200,24)
      integer win(2)            !   old code


     
       xname  ='x000000_00000000_00000'
       yname  ='y000000_00000000_00000'
       zname  ='z000000_00000000_00000'
       bname  ='b000000_00000000_00000'
       gname  ='g000000_00000000_00000'


      call system('rm channel.lis')
      open(unit=95,file='channel.lis',status='new')
       
      call system('rm ms.lis')
      open(unit=3,file='ms.lis',status='new')
c ------- Lecture des parametres et construction des listes -------
      print *,'call readpar'
      call readpar
      
      sundec=0
      uint=0
      ipermu=1
      ijcam=1536

c ------- Bornes de sequences (dark x, flat y, stop z, obs b) -------
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

c ------- Dimensions CCD et permutation (isp,jsp) -------
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
      
 1    format(38a)
 5    format(20i5)
c ================================================
c Boucle principale : moyennes dark (nxy=1) puis flat (nxy=2)
c ================================================
      do 300 nxy=1,2            !4
               write(3,*)' '
      write(3,*)' Loop nxy ',nxy
         print *,' '
         print *,' nxy,nfa,nfb ',nxy,nfa(nxy),nfb(nxy)
         
      print *,' '
      if(nxy.eq.1)then
         print *,' DARK'
         call system('ls m*x1.fit > xtab.lis')
         print 1,' xtab.lis'
         open(unit=11,status='old',file='xtab.lis')
      endif
      if(nxy.eq.2)then
         print *,' FLAT'
         call system('ls m*y1.fit > ytab.lis')    !     y -> b
         print 1,' ytab.lis'
         open(unit=12,status='old',file='ytab.lis')
      endif
      
      do 100 nf=1,nfb(nxy)
      if(nxy.eq.1)read(11,'(a)',iostat=ier,end=100) file(nxy,nf)
      if(nxy.eq.2)read(12,'(a)',iostat=ier,end=100) file(nxy,nf)
 6    format(' nf file(nxy,nf) ',i4,2x,a38)
      write(3,6)nf,file(nxy,nf)
      
      
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
 3    continue
 100  continue      ! nfb(nxy)
      write(3,*)' nxy,  useful nfiles(nxy)= ',nxy,nfiles(nxy)

         if(nxy.eq.1)rewind(11)
         if(nxy.eq.2)rewind(12)
         
         write(3,*)' '
         write(3,*)' Average nxy ',nxy
         write(3,*)' from file ',nfa(nxy),' to ',nfb(nxy)
      do j=1,jsp
         do i=1,isp
            tabaver(i,j)=0
         enddo
      enddo
      iu=20+nxy
      do 200 nf=nfa(nxy),nfb(nxy) !1,nfiles(nxy)
         write(3,*)' nxy,nf,file(nxy,nf) ', nxy,nf,file(nxy,nf) 
         ktab=0
         if(nf.eq.nfa(nxy))ktab=1
 4       format('head: ',a500)
            print *,' file(nf) ',file(nxy,nf)
c ------- Lecture et sommation des fichiers de la sequence -------
         call openold38(file(nxy,nf),sundec,iu)
         call counthead(iu,nhead,chead)
      
      if(nf.eq.nfa(nxy))then   
      print *,' nhead= ',nhead
         write(3,4)chead
         print 4,chead
      endif
           write(3,*)' readfits ipermu=',ipermu
      call readfits(iu,nhead,iswap,is,js,ipermu,tab2,tabpermu,ktab)
      write(3,*)' nxy=',nxy,'  file(',nf,') '
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
      print *,' AVERAGE'
      if(nxy.eq.1)print *,'xname  ',xname
      if(nxy.eq.2)print *,'yname  ',yname
      print *,' isp,jsp ',isp,jsp            !   avant erreur ?

      kpiv=nfb(nxy)-nfa(nxy)
      denom(nxy)=float(kpiv)+1.
      write(3,*)' nxy, denom(nxy) ',nxy,denom(nxy)
      do jp=1,jsp
      do ip=1,isp
         tab2(ip,jp)= float(tabaver(ip,jp))/denom(nxy) +0.5
      enddo
      enddo
 7    format(15i5)
 9    format(i4,2x,15I5)
 11   format(' ',a22,'    denom=',f6.3,'  extreme points  ', 4i5)
      head(1)=3
      head(2)=isp    !1024
      head(3)=jsp    !1536
      head(4)=1
      do n=4,512
         head(n)=0
      enddo

      print *,' write files xname,yname '
      iut=30+nxy
         call system('rm '//name(nxy))
       open(unit=iut,file=name(nxy),
     1      form='unformatted',status='new')
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
  
      write(3,11)name(nxy),denom(nxy),
     1        tab2(1,1),tab2(isp,1),tab2(isp,jsp),tab2(1,jsp)
      
      do j=1,jsp,100
      write(3,9)j,(tab2(i,j),i=1,isp,100)
      enddo
 300  continue                 ! end loop nxy
      close(11)
      close(12)
      close(14)
      close(21)
      close(22)
      rewind(31)
      rewind(32)
      rewind(33)      
      nw=1
      win(1)=1
      win(2)=0
      nm=9
      

      imb=1536
      jmb=1024
      imc=1024
      jmc=1536
      nm=9
      write(3,*)' enter geom'
c ================================================
c Appel du calcul de geometrie (ms2.f) sur les fichiers moyens
c ================================================
      call geom(nw,win,nm,31,32,32,gname,istop,ima,ijcam,imima,jmima,
     1                                              xr,yr,imc,jmc)
      write(3,*)' end geom'

      
      close(unit=95)
      close(unit=3)
           
      stop
      end
c ================================================
c readpar : lecture 'balayage' de ms.par (sans stocker)
c ================================================
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
c ================================================
c par1 : extraction d'un parametre nomme de ms.par
c ================================================
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
c ================================================
c readfits : lecture FITS brute + permutation
c ================================================
      Subroutine readfits
     1     (iu,inbhead,iswap,is,js,ipermu,tab2,tabpermu,ktab)
       integer*2 tab2(1536,1536)
       integer*4 tabpermu(1536,1536)
       integer*2 sort(1440),in(1),out(1),ku
       integer t,uint
       write(3,*)' readfits iu,inbhead,is,js,ipermu,ktab',
     1                     iu,inbhead,is,js,ipermu,ktab   
        n=inbhead
        i=0
        j=1
        k=1
 100    n=n+1
        read(iu,rec=n,iostat=ios)  (sort(t),t=1,1440)
        
        if(ios.lt.0) go to 1000
        t=1
 200    i=i+1

        if(i.le.is) go to 500
         if(j.lt.js) then

           j=j+1               !   loop j
           i=1
         else
          goto1000
          endif

 500      continue
            lswap=0
            if(i.eq.is/2.and.j.eq.js/2)lswap=1
          if(iswap.eq.1)then
              in(1)=sort(t)
              call swap(in,1,out,lswap)
              sort(t)=out(1)
          endif
          tab2(i,j)=sort(t)     ! before prmut

       t=t+1
       if (t.eq.1441) go to 100
       go to 200                   !   loop i
1000   continue
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
       print *,' tab2 before permut'
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


c ================================================
c swap : inversion d'octets (big->little endian)
c ================================================
       subroutine swap(in,ncar,out,lswap)   
      integer*2 in(1),out(1),auxi2
      logical*1 low,auxl1(2)
      equivalence (auxi2,auxl1(1))
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
c ================================================
c counthead : compte le nb de blocs d'en-tete
c ================================================
      subroutine counthead(iu,nb,chead)
      integer nb,head(512)
      character chead*2880
               write(3,*)' subroutine counthead'
      do nb=1,10
      read(iu,rec=nb) chead(1:2880)

      i=1
         do n=1,36 
         if(chead(i:i+3).eq.'END ') go to 1000
         i=i+80
         enddo
      enddo
 1000 continue
      return
      end
c ================================================
c openold38 : ouverture en acces direct
c ================================================
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
c ================================================
c Routines d'ouverture sequentielle 22 caracteres
c ================================================
      subroutine opennew22(xyname,iu)
      character*22 xyname
      open(unit=iu,file=xyname,status='new')
      return
      end
      subroutine openold22(name,sundec,iu)
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='old',form='unformatted',file=name)
        write(3,*)'ouvert'
      return
      end
      subroutine opennew22sf(name,sundec,iu)
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='new',form='formatted',file=name)
      return
      end
c ================================================
c comptehead : variante avec journalisation
c ================================================
            subroutine comptehead(iu,nb)     !  double counthead????
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
      subroutine openold22sf(name,sundec,iu)
      character*22 name
      integer sundec
        write(3,*)name
      open(unit=iu,status='old',form='formatted',file=name)
      return
      end
c ================================================
c DPMCAR : moindres carres double precision
c ================================================
      subroutine DPMCAR(X,Y,P,ND,NT,COEF)
      DOUBLE PRECISION C,F,PIVOT
      DIMENSION X(2000),Y(2000),P(2000),COEF(10),F(10),C(11,11)
      IF(NT.GT.10)NT=10
      NTP=NT+1
      DO 2 I=1,NT
      DO 2 L=1,NTP
2     C(L,I)=0.
      DO10 L=1,ND
      F(1)=1
        DO7 I=2,NT
7       F(I)=F(I-1)*X(L)/1000.
      DO10 M=1,NT
        DO9 K=M,NT
9       C(K,M)=C(K,M)+F(K)*F(M)*P(L)
10    C(NTP,M)=C(NTP,M)+F(M)*P(L)*Y(L)
      DO20 M=1,NTP
      DO20 K=M,NTP
20    C(M,K)=C(K,M)
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
      COEF(1)=C(NTP,1)
      DO106 I=2,NT
106   COEF(I)=C(NTP,I)/(1000.**(I-1))
      RETURN
      END
