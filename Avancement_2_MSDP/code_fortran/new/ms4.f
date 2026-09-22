c     ms4.f
      subroutine ivmap4(sobstot,cal,iim,jjm,nm,nfb1,nfb2,profnf)
c     maps I/V                885 123  9          interpolations degree 4
      dimension sobstot(2000,200,24,10),cal(2000,200,24),P(2000)
c                                nm
      dimension profnf(2000,200,250,3),prof(250),prof5(250),X5(100)
c                       x    y  nlm interpn  nf
      dimension velint(200,2000,3,2,10)
c                       x    y  bisect velint nfb
      dimension x(2000),y(2000),z(250),COEF(10),nl(250 ),yl(250)
c                                        abs 1  1.1  1.2 / ordonnées profnf     
      dimension xx(250),yy(250),zz(250),pro(10,100),pro81(81)
      dimension profic(81,3,3),proplot(81),pro1(81),pro2(81),pro3(81)
c                    nfm nfig ideg
c     for   nlm=1+10*8 jfig=21,61,101   ideg=1,2,3   (profiles in i=icc)

  4   format(2i8,2f8.2)      !   for miv.lis
c      write(4,*)'   intvel  lbdvel     xv2     yv2'
      
      interp=3
      ideg=1                    ! (deg 3)
      nfb=1
      nfm=(nm-1)*10+1           !81
      iic=(iim+1)/2

      do jj=21,jjm,40
         do nf=1,nfm  
            profnf(iic,jj,nf,3)=profnf(iic,jj,nf,1)  ! 3=mix with interp deg 3
         enddo
      write(3,*)' profnf(iic,jj,m,1)  in  ms4  jj=',jj
 3    format(11F7.0)
      write(3,3)(profnf(iic,jj,m,1),m=1,81,10)  !   begin
      enddo
      write(3,*)'   '
c**********************************      profic   ideg=1
      nfig=1     !  deg 3
      do nfig=1,3
         jfig=21+(nfig-1)*40
         do nf=1,nfm    !  81
            profic(nf,nfig,1)=profnf(iic,jfig,nf,1)
         enddo
      enddo
 1    format(11F7.0)
      do nfig=1,3
      write(3,*)' profic(nf,nfig,1)   nfig= ',nfig
      write(3,1) (profic(nf,nfig,1),nf=1,nfm,10)
      enddo
 
c*********************
      nfb2=1
      print *,' ivmap 4'
      write(3,*)' ivmap4  iim jjm nm',iim,jjm,nm
c--------------------------------
c  Lines profiles interpolation   (1+(nm-1)°4 lambdas)
c     *************
c      nlm=4*nm-1     ! new lambdas
c      ND=4
c      NT=4
c      call par1('    nfb1',1,nfb1)
c      call par1('    nfb2',1,nfb2)
      nfb1=1
      nfb2=1
      do  k=1,9
         P(k)=1.                !  poids
      enddo
      write(3,*)' ivmap4 boucles nfb,i,j,nm'
      write(3,*)' ivmap4 iim jjm',iim,jjm
c-----------------------------------------------------------
c      do 20 n=1,nm
c         nx=n
c         xl(nx)=1.+nx-1       !  lambda
c 20   continue
c      write(3,*)' ivmap xl(n) ',(xl(n),n=1,nm)
      write(3,*)' ivmap boucles nfb,i,j,nm'
      print *,' ivmap 3'
      iic=(iim+1)/2
      jjc=(jjm+1)/2
      ntm=5
c**********************

      write(3,*)' ivmap4 iic jjc nfb1 nfb2 ',iic,jjc,nfb1,nfb2
 5    format(20f6.0)      
      write(3,*)' sobstot(170,j,5,1) '
      write(3,5)(sobstot(170,j,5,1),j=1,jjm,5)
      write(3,*)' sobstot(iic,61,n,1) '
      write(3,5)(sobstot(iic,61,n,1),n=1,9)
      
c      do 100 nfb=nfb1,nfb2 
c         print *,' nfb=',nfb
c         write(3,*)' ivmap4 iim jjm iic jjc nfb',iim,jjm,iic,jjc,nfb !oui

         i=iic         !iic
            do 80 j=21,jjm,40    !1,jjm,10     
               write(3,*)' boucle j= ',j ! oui    21               
c     write(3,*)'  DPMCAR ' !  Double Précision Moindres Carré
      ntm=5
c      nf1=10*nt1-9   !  11
c      nf2=10*nt2-9              !   51
c-------
      if(j.eq.21)then
      n0=2
      endif
      if(j.eq.61)then
      n0=3
      endif
      if(j.eq.101)then
      n0=4
      endif
c      do 70 n0=1,5      !n=3,nm-4            !   3 - 5                  
      write(3,*)' j n0 ntm ',j,n0,ntm   !  oui

      
        do 72  k=1,ntm            !    1   6    2-7    3-9
                     ny=k+n0-1      !k+n0-3         !  1 - 6     2-8
c                     nf=1+10*(k-1+nt1)   
                     xx(k)=float(k)        !     
                     yy(k)=sobstot(i,j,ny,nfb1) ! 
 72             continue
          call DPMCAR(xx,yy,P,ntm,ntm,COEF)                 
          do 73 k=1,ntm
c     nf=1+10*(k-1)  !        1 11  ..  81
c     nf=1+10*(k-1+nt1)   !
                 ny=n0+k-1  !nt1+k-1   !  2
c                 nf1=1+10.*(nt1) ! 11    51
c                 nf2=1+10.*(nt2)
                     xx(k)=float(k)   
          zz(k)= COEF(1)+xx(k)*(COEF(2)+xx(k)*(COEF(3)+xx(k)*(COEF(4)+
     1            +xx(k)*(COEF(5)))))      !)+xx(k)*(COEF(6))))))
 73    continue
 76    format(5f8.0)
c77    format(9f8.3)
c      if(j.eq.21)then
       write(3,*)' DPMCAR COEF j=',j
       write(3,*)(COEF(kc),kc=1,ntm)
c       do k=1,9
c          nf=1+10*(k-1)  !  1 11  ..  81 
c          xx(k)=float(k)
c       enddo
       write(3,*)' DPMCAR xx yy zz'
       write(3,76)(xx(k),k=1,ntm),(yy(k),k=1,ntm),  !oui
     1            (zz(k),k=1,ntm)
c      endif
      
c----------------------------calcul de 5 points pour profnf 
c         sobstot(i,j,n,nfb)                  
c  data  n               1    2    3    4    5    6    7    8    9    channels
c  lambdas DPMCAR  nd    0   10   20   30   40   50   60   70   80    data
c  indices nouveaux nf   1   11   21   31   41   51   61   71   81    profnf
c              n0             2                                                c
c            xx(k)            1.                  5.
c              n0                  3                   
c     calculs    n=      1---------3****4         6
c                             2         4****5         7
c                                  3         5****6         8        
c                                       4         6****7---------9

c                nf    nf1       nf2  nf3       nf4        
c                k       1         3    4         6
c                xk     1.0       3.0  4.0       6.0             
c     résultats profnf(i,j,nf,nfb) 
c     abscisses   x       0.       20.  30.       50.                 
c----- ----------------------------------------------------------
             nf1=(n0-1)*10+1 !11   21   31
             nf2=(n0+3)*10+1    !51   61   71
             nff1=nf1
             nff2=nf2  
      do 44 nff=nff1,nff2 ! 11  12  13 ..  21....      ... 51
      x(nff)=1.         +float(nff-nff1)/10. 
      pro(n0,nff)=COEF(1)+
     1        x(nff)*(COEF(2)+x(nff)*(COEF(3)+x(nff)*(COEF(4)+
     2         x(nff)*(COEF(5)))))
      profnf(iic,j,nff,2)=pro(n0,nff)   !  ivprof2
      profnf(iic,j,nff,3)=pro(n0,nff)   !  pro(n0,nff)    ! ivprof3   (begin)
c                                          *******            
 44   continue
 45       format(10f6.0)
      write(3,*)' DPMCAR  pro(n0,nff)  j  n0  nff1,nff2 ',j,n0,nff1,nff2
      write(3,45)(pro(n0,nff),nff=nff1,nff2,10)
c      write(3,*)' profnf  j=',j
 74   format(10f6.0)
c      write(3,74)(profnf(iic,j,nff,1),nd=1,81)
c---------------------------------------  
 80        enddo ! j
c     100  enddo                     ! nfb
c=============================================   plots interp 2 and 3
           nlm=81
c           nfbb=nfb1 
       call par1(' lbdvel1',1,lbdvel1)
       call par1(' lbdvel2',1,lbdvel2)
c********************************************    interp=2,3    ivprof2 begin3
       call pgbegin(0,'ivprof2.ps/ps',3,1) 
       call pgsch(2.5) !1.8)
c       call pgslw(2)
       iic=(iim+1)/2
c       jjc=(jjm+1)/2
c      call pgvport(0.1,0.9,0.3,0.7)   !0.3,0.7)
      call pgslw(3)
c      call pgsch(2.5)
c---------
c 210  format(10f6.0)
      do 200 jj=21,101,40
         write(3,*)'                                 profils  jj'
      if(jj.eq.21)then
      n0=2
      endif
      if(jj.eq.61)then
      n0=3
      endif
      if(jj.eq.101)then
      n0=4
      endif
      if(jj.eq.21)intmin=4
      if(jj.eq.61)intmin=5
      if(jj.eq.101)intmin=6
      ntm=5
      nff1=10*(n0-1)+1
      nff2=nff1+40
      nf1=nff1
      nf2=nff2
      write(3,*)' profils jj n0 ntm  nff1 nff2',jj,n0,ntm,nff1,nff2
      
      call pgadvance
      call pgvport(0.3,0.9,0.3,0.7)
      call pgsch(2.5)
      if(jj.eq.21)call pglabel('Channels','Intensity',
     1             'Profile in X=20')
      if(jj.eq.61)call pglabel('Channels','Intensity',
     1     'Profile in X=60')
      if(jj.eq.101)call pglabel('Channels','Intensity',
     1             'Profile in X=100')
      call pgwindow(9.,1.,0.,2000.)
      call pgbox('bcints',1.,0,'bcints',500.,5) !   1 2 3 ...9

c      write(3,*)' fig profnf iic jj nf1 nf2 nfb1',iic,jj,nf1,nf2,nfb1
c     write(3,210)(pro(n0,nff),nff=nff1,nff2,10)! (pro(iic,jj,m,nfb1),m=nf1,nf2)
c      do m=1,81     
c         xx(m)=float(m)-1
c         mp=m-1
c         yy(m)=pro81(mp)         
c      enddo
c-----------------------------------         
      maxmp=nff2-nff1+1 !41
      do 211 mp=1,maxmp
         nff=nff1+mp-1
         prof5(mp)=pro(n0,nff)         !     prof5
        x5(mp)=nff1+mp-2
 211  continue   
      call pgwindow(80.,0.,0.,2000.)
         call pgline(41,x5,prof5) !   nf2,x5,prof5)            !profil
         write(3,*)' x5 prof5 maxmp nf1 nf2',maxmp,nf1,nf2
         write(3,*)(x5(mp),mp=1,41,10)
         write(3,*)(prof5(mp),mp=1,41,10)
c--------------------------------------------
      call pgwindow(9.,1.,0.,2000.)
      do m=1,9    
         xxp=float(m)
         yyp=sobstot(iic,jj,m,1)    !profnf(iic,jj,m,nfb1)
         call pgsch(3.)
       call pgpoint(1,xxp,yyp,17)       !  points
       call pgsch(1.8)                                        !  modif
      enddo
c      write(3,*)'intvel4 iic jj nm nfb1 lbdvel nlm ',
c     1     iic,jj,nm,nfb1,lbdvel,nlm
c     if(lbdvel.ne.0)then
      
c      goto 198
      call pgwindow(80.,0.,0.,2000.)
      lbdvel=lbdvel1
      call pgslw(4)
      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)   !,xv2,yv2)
c           write(4,4) 4,lbdvel1,xv2,yv2   
      call pgslw(3)
      lbdvel=lbdvel2
      call pgsls(4)
      call pgslw(6)
      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)   !,xv2,yv2)
c              write(4,4) 4,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
c         call intvel5(jj,prof5,320,nf1,nf2,nlm)         
c     endif
c 198  continue
 200  continue                          !  j
      call pgend
      call par1(' ivprof2',nw,ivprof2)
      if(ivprof2.eq.1)call system('okular ivprof2.ps &')
c***************************************plot      interp 2,3  ivprof3
      call pgbegin(0,'ivprof3.ps/ps',3,1) ! deg 3 and 4
       iic=(iim+1)/2
      call pgvport(0.1,0.9,0.3,0.7)   !0.3,0.7)
      call pgslw(3)
      call pgsch(2.5)
c---------
            do 400 jj=21,101,40
         write(3,*)'                                 profils  jj'
      if(jj.eq.21)then
      n0=2
      endif
      if(jj.eq.61)then
      n0=3
      endif
      if(jj.eq.101)then
      n0=4
      endif
      if(jj.eq.21)intmin=4
      if(jj.eq.61)intmin=5
      if(jj.eq.101)intmin=6
      ntm=5
      nff1=10*(n0-1)+1
      nff2=nff1+40
      nf1=nff1
      nf2=nff2
      write(3,*)' profils jj n0 ntm  nff1 nff2',jj,n0,ntm,nff1,nff2
      
      call pgadvance
      call pgvport(0.3,0.9,0.3,0.7)    
            if(jj.eq.21)call pglabel('Channels','Intensity',
     1             'Profile in X=20')
      if(jj.eq.61)call pglabel('Channels','Intensity',
     1     'Profile in X=60')
      if(jj.eq.101)call pglabel('Channels','Intensity',
     1             'Profile in X=100')
      call pgwindow(9.,1.,0.,2000.)
      call pgbox('bcints',1.,0,'bcints',500.,5) !   1 2 3 ...9

      call pgwindow(9.,1.,0.,2000.)
      do m=1,9    
         xxp=float(m)
         yyp=sobstot(iic,jj,m,1)    !profnf(iic,jj,m,nfb1)
         call pgsch(3.)                  !  modif
       call pgpoint(1,xxp,yyp,17)                            !  points
       call pgsch(1.8)                                !  modif
      enddo

      do m=1,81
         mp=m-1
         xx(m)=float(mp)
         yy(m)=profnf(iic,jj,m,3)                            ! image
      enddo
      call pgwindow(80.,0.,0.,2000.)
      call pgline(81,xx,yy)

      call par1(' lbdvel1',nw,lbdvel1)
      call par1(' lbdvel2',nw,lbdvel2)
      call par1(' lbdvel3',nw,lbdvel3)
      write(3,*)' nlm yy ',nlm,(yy(nf),nf=1,81,10)
      if(lbdvel1.ne.0)then
               write(4,*)'   intvel  lbdvel     xv2     yv2'
               call intvel3(yy,lbdvel1,nlm,xv2,yv2)
               v2km=xv2/vinterp
         write(4,4) 3,lbdvel1,xv2,yv2
      endif      
      if(lbdvel2.ne.0)then
         call pgsls(4)
         call pgslw(6)
         call intvel3(yy,lbdvel2,nlm,xv1,yv2)
                 write(4,4) 3,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
      endif
      if(lbdvel3.ne.0)then
      call intvel3(yy,lbdvel3,nlm,xv2,yv2)
      write(4,4) 3,lbdvel3,xv2,yv2
      endif

c                                             points A,B
      yyy=profnf(iic,jj,nff1,3)
      xxx=nff1-5      !xl/2.-trj2-9.
      call pgtext(xxx,yyy,'B')
      yyy=profnf(iic,jj,nff2,3)
      xxx=nff2-5        !xl/2.-trj2-9.
      call pgtext(xxx,yyy,'A')
 400  continue

c      goto 500
c            call pgwindow(80.,0.,0.,2000.)
c      lbdvel=lbdvel1
c      call pgslw(4)
c      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)   
c      call pgslw(3)
c      lbdvel=lbdvel2
c      call pgsls(4)
c      call pgslw(6)
c      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)
c      call pgsls(1)
c      call pgslw(3)
c 500  continue

      call pgend
      call par1(' ivprof3',nw,ivprof3)
      if(ivprof3.eq.1)call system('okular ivprof3.ps &')
c*********************************
      return
      end
c***********************************************
c-------------------------velocity    lbdvel = distance en lambdas cubiques
      subroutine intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)     !,xv1,yv2)
      dimension  diff(250),ddiff(250),pro(10,100),prof(250)
      dimension xvert(2),yvert(2),xhor(2),yhor(2)
c   data  n                1    2    3    4    5    6    7    8    9    channels
c   lambdas DPMCAR  nd     0   10   20   30   40   50   60   70   80    data
c   indices nouveaux nf,nl 1   11   21   31   41   51   61   71   81    profnf

c                            B2                         A1
c                                B        ><        A
c                                   B1         A2
c                     N     nl+31  nl+30      nl+1       nl              
c                     X     nl+30  nl+29      nl         nl-1         
c
c         (XA-XA1)/(XA2-XA1)=(yA-YA1)/(YA2-YA1)  ->  XA
      nf1=nff1
      nf2=nff2
      write(3,*)' sub intvel5  jj lbdvel nf1 nf2 nlm ',
     1    jj,lbdvel,nf1,nf2,nlm
c****************************************************
      do 3 nl=1,nlm
         prof(nl)=pro(n0,nl)
 3    continue
      write(3,*)' intvel  prof '
      write(3,5)(prof(nl),nl=nf1,nf2)     ! oui
      nlmb=nf2-lbdvel          !-1
      do 4 nl=nf1,nlmb
         diff(nl)=prof(nl+lbdv)-prof(nl)  
 4    continue
      write(3,*)' intvel  diff jj=',jj
 5    format(10f6.0)
      write(3,5)(diff(nl),nl=nf1,nlmb) !  oui
         nlmb1=nlmb-1
      do 10 nl=nf1,nlmb1
         nla1=nl
         nla2=nl+1  
         nlb1=nl+lbdvel
         nlb2=nl+1+lbdvel
         diff(nla1)=prof(nlb1)-prof(nla1)
         diff(nlb1)=prof(nlb2)-prof(nla2)
         write(3,*)' nl diff(nla1) diff(nlb1) ',nl,diff(nla1),diff(nlb1)  !oui
        if(diff(nla1)*diff(nlb1).le.0)goto 15
 10   continue
 15   continue
           dn=-diff(nla1)/(diff(nlb1)-diff(nla1))
           write(3,*)' intvel4 diff jj  nla1 dn',jj,nla1,dn   ! oui
            xl=float(nla1)+dn+float(lbdvel)/2.-1.
            yl=prof(nla1)+dn*(prof(nlb1)-prof(nla1))     
            write(3,*)' diff(nla1) diff(nlb1) dn xl yl ',
     1           diff(nla1),diff(nlb1),dn,xl,yl  

      xvert(1)=xl  !float(nla)+dn-1.+float(lbdvel)/2.    !  x=index-1
      xvert(2)=xl  !  -1=intervalle cubique
      yvert(1)=0.
      yvert(2)=yl
      xhor(1)=float(nla1)+dn-1.
      xhor(2)=xhor(1)+float(lbdvel) !    float(nla)+dn+float(lbdvel)-1.
      yhor(1)=yl
      yhor(2)=yl
      write(3,*)' xhor 1 2 yhor 1 2',xhor(1),xhor(2),yhor(1),yhor(2)
      write(3,*)' xvert 1 2 yv 1 2',xvert(1),xvert(2),yvert(1),yvert(2)
      xv2=xvert(2)
      yv2=yvert(2)
      write(3,*)' '
      write(3,*)' intvel4,lbdvel,xv2,yv2',
     1                  4,lbdvel,xv2,yv2
      call pgline(2,xhor,yhor)
      call pgline(2,xvert,yvert)
      call pgsls(1)
      return
      end
c================================================
c---------------------------------------
c      subroutine DPMCAR(X,Y,P,ND,NT,COEF)
c                ------
c Ce sous-programme [ Schneider 75 ] calcule par une 
c methode de moindres carres en double precision le 
c polynome de NT termes (10 max) Y=F(X) associe a ND mesures.
C
C P=poids
      
      
