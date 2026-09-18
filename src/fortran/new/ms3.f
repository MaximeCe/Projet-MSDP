c      ms3.f
      subroutine ivmap3(sobstot,cal,iim,jjm,nm,nfb1,nfb2,profnf)
c                intensity/velocity  interpolations degree 3
c     maps I/V                885 123  9
      dimension sobstot(2000,200,24,10),cal(2000,200,24),P(2000)
c                                nm
      dimension profnf(2000,200,250,3),prof(250)
c                       x    y  nlm nf
      dimension x(2000),y(2000),z(250),COEF(4),  nl(250),yl(250)
c                                        abs 1  1.1  1.2 / ordonnées profnf     
      dimension xx(250),yy(250)
 4    format(2i8,3f8.2)         !   for miv.lis
      write(4,*)'   intvel  lbdvel     xv2     yv2   xv2 km/s'
 
      nfb1=1
      nfb2=1
      print *,' ivmap 3'
      write(3,*)' ivmap3  iim jjm nm',iim,jjm,nm
c  Lines profiles interpolation   (1+(nm-1)°4 lambdas)
c     *************
c-------------------------------
c      nfb=1
c      call calobs3(cal,sobstot,iim,jjm,nm,nfb)
      iic=(iim+1)/2
      jjc=(jjm+1)/2
      ntm=5
     
      write(3,*)' ivmap3  sobstot(iic,jjc,5,1) ',sobstot(iic,jjc,5,1)
c----------------------------------------------      
c      nlm=4*nm-1     ! new lambdas
c      ND=4
c      NT=4
      call par1('    nfb1',1,nfb1)
      call par1('    nfb2',1,nfb2)
      nfb2=1
      
      do  i=1,4
         P(i)=1.                !  poids
      enddo
      write(3,*)' ivmap3 boucles nfb,i,j,nm'
      write(3,*)' ivmap3 iim jjm',iim,jjm
c-----------------------------------------------------------
c      do 20 n=1,nm
c         nx=n
c         xl(nx)=1.+nx-1       !  lambda
c 20   continue
c      write(3,*)' ivmap xl(n) ',(xl(n),n=1,nm)
      write(3,*)' ivmap3 boucles nfb,i,j,nm'
      print *,' ivmap 3'
      iic=(iim+1)/2
      jjc=(jjm+1)/2
      write(3,*)' ivmap iic jjc',iic,jjc
    
      do 100 nfb=nfb1,nfb2
         print *,' nfb=',nfb
         write(3,*)' ivmap3 iim jjm iic jjc nfb',iic,jjm,iic,jjc,nfb
         do 90 i=iic,iic
c            jjc=(jjm+1)/2
            do 80 j=1,jjm,10     
      write(3,*)' boucle j= ',j
c     write(3,*)'  DPMCAR ' !  Double Précision Moindres Carrés 
           do 70 n=2,nm-2         !   1  7    
                  do 72 k=1,4
                     np=k+n-2      !  1  2  3  4
                     x(k)=10.*float(np-1) !  0.  10.  20.  30.
                     y(k)=sobstot(i,j,np,nfb1) !     n=2 ->  1  2  3  4
 72             continue
          call DPMCAR(x,y,P,4,4,COEF)                 
          do 73 k=1,4
            z(k)= COEF(1)+x(k)*(COEF(2)+x(k)*(COEF(3)+x(k)*COEF(4)))
 73      continue
      write(3,*)' DPMCAR n y z',n,(x(k),k=1,4),(y(k),k=1,4),(z(k),k=1,4)
c   ----------------------------calcul de 10 points pour profnf entre n et n+1
c         sobstot(i,j,n,nfb)                  
c  data  n               1    2    3    4    5    6    7    8    9    channels
c  lambdas DPMCAR  nd    0   10   20   30   40   50   60   70   80    data
c  indices nouveaux nf   1   11   21   31   41   51   61   71   81    profnf
c  calculs    n= 1       1*********    31            
c                2           11   *******   41
c                3                21-----*****---51
c                                               
c                7                               51    *********81  
c    résultats profnf(i,j,n,nfb)
c-----------------------------------------------------------------        
      if(n.eq.2)then
        nd1=0
        nd2=20
      endif
      if(n.gt.2.and.n.lt.nm-2)then
         nd1=10*(n-1)  !  2
         nd2=nd1+10
      endif
      if(n.eq.nm-2)then
         nd1=(nm-3)*10    ! 2?
         nd2=nd1+20
      endif

          do nd=nd1,nd2   
             x(nd)=float(nd)    !
             nf=nd+1
             profnf(iic,j,nf,1)=   !  11  12  13 ... 21    pour n=2
     1         COEF(1)+x(nd)*(COEF(2)+x(nd)*(COEF(3)+x(nd)*COEF(4)))
          enddo
c          nd1=10*(n-1)+1     !  11....
c     nd2=10*(n-1)+11        !    .... 21
 70   continue
      write(3,*)' profnf  j=',j
 74   format(10f6.0)
      write(3,74)(profnf(iic,j,nd,1),nd=1,81)
c---------------------------------------
 80        enddo ! j
 90        enddo ! i
 100    enddo   ! nfb        
c       nlm=81
c       j=21
c          write(3,*)'1,iic,j,profnf ',1,iic,j
c110       format(10f8.0)
c          j=21
c          do m=1,nlm
c             prof(m)=profnf(iic,j,m,1)
c          enddo
c          write(3,110)(prof(m),m=1,nlm)
c     write(3,*)' '
        write(3,*)'   '
      do jj=21,jjm,40
      write(3,*)' profnf(iic,jj,m,1)  out  ms3  jj=',jj
 150  format(11F7.0)
      write(3,150)(profnf(iic,jj,m,1),m=1,81,10)
      enddo
c=============================================
           nlm=81
c         nfbb=nfb1
           call par1(' lbdvel1',1,lbdvel1)
           call par1(' lbdvel2',1,lbdvel2)
           call par1(' lbdvel3',1,lbdvel3)
       call pgbegin(0,'ivprof1.ps/ps',3,1)
       call pgsch(2.5) !1.8)
       call pgslw(3)
       iuc=(iim+1)/2
       juc=(jjm+1)/2

c      call pgvport(0.1,0.9,0.3,0.7)   !0.3,0.7)
      call pgslw(3)
      call pgsch(2.5)
c                                 image 1
      call pgadvance
      call pgvport(0.3,0.9,0.3,0.7)
      call pglabel('Channels','Intensity',
     1             'Profile in X=20')
      call pgwindow(9.,1.,0.,2000.)
      call pgbox('bcints',1.,0,'bcints',500.,5) !   1 2 3 ...9

      jj =21
      do m=1,81               !  new lambdas  nl=1   x=90
            xx(m)=float(m-1)
            yy(m)=profnf(iic,jj,m,1)
      enddo
         call pgwindow(80.,0.,0.,2000.)
         call pgline(81,xx,yy)            !profil

      do m=1,81,10    
         xxp=float(m-1)
         yyp=profnf(iic,jj,m,1)
         call pgsch(3.)
       call pgpoint(1,xxp,yyp,17)       !  points
       call pgsch(1.8)
      enddo
c*****
      write(3,*)' ms3 profnf(iic,101,m,1)',
     1               (profnf(iic,101,m,1),m=1,81,10)
c*****      
      write(3,*)'intvel3 iic jj nm nfb1 lbdvel nlm ',
     1     iic,jj,nm,nfb1,lbdvel,nlm
      if(lbdvel1.ne.0)then
         call intvel3(yy,lbdvel1,nlm,xv2,yv2)
         write(4,4) 3,lbdvel1,xv2,yv2
      endif
      if(lbdvel2.ne.0)then
      call pgsls(4)
      call pgslw(6)
      call intvel3(yy,lbdvel2,nlm,xv2,yv2)
               write(4,4) 3,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
      endif
      if(lbdvel3.ne.0)then
         call intvel3(yy,lbdvel3,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel3,xv2,yv2
      endif
c                                    image 2
      call pgadvance
      call pgvport(0.3,0.9,0.3,0.7)
      call pgsch(2.5)
      call pglabel('Channels','Intensity',
     1             'Profile in X=60')
      call pgwindow(9.,1.,0.,2000.)
      call pgbox('bcints',1.,0,'bcints',500.,5) !   1 2 3 ...9
 
      jj =61
      do m=1,81               !  new lambdas  nl=1  
            xx(m)=float(m-1)
            yy(m)=profnf(iic,jj,m,1)
      enddo
         call pgwindow(80.,0.,0.,2000.)
        call pgline(81,xx,yy)            !  profil

        do m=1,81,10
         xxp=float(m-1)
         yyp=profnf(iic,jj,m,1)
         write(3,*)' ivmap m xxp,yyp',m,xxp,yyp
         call pgsch(3.)
       call pgpoint(1,xxp,yyp,17)       !  points
       call pgsch(1.8)
      enddo
      if(lbdvel1.ne.0)then
c         call pgslw(4)
         call intvel3(yy,lbdvel1,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel1,xv2,yv2
c         call pgslw(3)
      endif
      if(lbdvel2.ne.0)then
         call pgsls(4)
         call pgslw(6)
         call intvel3(yy,lbdvel2,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
      endif
      if(lbdvel3.ne.0)then
c         call pgslw(4)
         call intvel3(yy,lbdvel3,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel3,xv2,yv2
c         call pgslw(3)
      endif
c                                    image 3
      call pgadvance
      call pgsch(2.5)
      call pgvport(0.3,0.9,0.3,0.7)
      call pglabel('Channels','Intensity',
     1             'Profile in X=100')
      call pgwindow(9.,1.,0.,2000.)
      call pgbox('bcints',1.,0,'bcints',500.,5) !   1 2 3 ...9      

      jj=101
      do m=1,81               !  new lambdas  nl=1   x=90
            xx(m)=float(m-1)
            yy(m)=profnf(iic,jj,m,1)
      enddo
         call pgwindow(80.,0.,0.,2000.)
        call pgline(81,xx,yy)              ! profil

       do m=1,81,10    
         xxp=float(m-1)
         yyp=profnf(iic,jj,m,1)
       call pgsch(3.)
       call pgpoint(1,xxp,yyp,17)     !  points
       call pgsch(1.8)
      enddo
      write(3,*)' intvel3 lbdvel ',lbdvel
      if(lbdvel1.ne.0)then
         call intvel3(yy,lbdvel1,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel1,xv2,yv2
      endif
      if(lbdvel2.ne.0)then
         call pgsls(4)
         call pgslw(6)
         call intvel3(yy,lbdvel2,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
      endif
      if(lbdvel3.ne.0)then
         call intvel3(yy,lbdvel3,nlm,xv2,yv2)
                  write(4,4) 3,lbdvel3,xv2,yv2
      endif
      call pgend
      call par1(' ivprof1',nw,ivprof1)
      if(ivprof1.eq.1)call system('okular ivprof1.ps &')

      return
      end
c***********************************************
c-------------------------velocity    lbdvel = distance en lambdas cubiques
      subroutine intvel3(yy,lbdvel,nlm,xv2,yv2)
      dimension  yy(250),diff(250),ddiff(250),pro(250)
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
c
      write(3,*)' sub intvel4 nlm lbdvel ',
     1     nlm,lbdvel      
c            call par1('  labsem',1,labsem)
c     
      do 3 nl=1,nlm
         pro(nl)=yy(nl)
 3    continue
      write(3,*)' intvel  pro '
      write(3,5)(pro(nl),nl=1,nlm)
      nlmb=nlm-lbdvel  !-1
      do 4 nl=1,nlmb
         diff(nl)=pro(nl+lbdvel)-pro(nl)
 4    continue
      write(3,*)' intvel3 diff '
 5    format(10f6.0)
      write(3,5)(diff(nl),nl=1,nlmb)
      
      nlmb1=nlmb-1
      do 10 nl=1,nlmb
         nla1=nl
         nla2=nl+1  
         nlb1=nl+lbdvel
         nlb2=nl+1+lbdvel
         diff(nla1)=pro(nlb1)-pro(nla1)
         diff(nlb1)=pro(nlb2)-pro(nla2)

        if(diff(nla1)*diff(nlb1).le.0)goto 15
 10   continue
 15   continue
           dn=-diff(nla1)/(diff(nlb1)-diff(nla1))
           write(3,*)' intvel3 diff  nla1 dd dn',nla1,dd,dn
            xl=float(nla1)+dn+float(lbdvel)/2.-1.
            yl=pro(nla1)+dn*(pro(nlb1)-pro(nla1))     
            write(3,*)' diff(nla) diff(nlb) dn xl yl ',
     1           diff(nla),diff(nlb),dn,xl,yl  
            
      xvert(1)=xl  !float(nla)+dn-1.+float(lbdvel)/2.    !  x=index-1
      xvert(2)=xl  !  -1=intervalle cubique
      yvert(1)=0.
      yvert(2)=yl
      xhor(1)=float(nla1)+dn-1.
      xhor(2)=xhor(1)+float(lbdvel) !    float(nla)+dn+float(lbdvel)-1.
      yhor(1)=yl
      yhor(2)=yl
      write(3,*)' lbdvel, xhor 1 2 yhor 1 2',
     1            lbdvel,xhor(1),xhor(2),yhor(1),yhor(2)
      write(3,*)' lbdvel, xvert 1 2 yv 1 2',
     1     lbdvel,xvert(1),xvert(2),yvert(1),yvert(2)
      write(3,*)' '
      xv2=xvert(2)
      yv2=yvert(2)
      write(3,*)' intvel3,lbdvel,xvert(1),yvert(2) ',
     1     3,lbdvel,xv2,yv2
      xv2=xvert(2)
      yv2=yvert(2)
      call pgline(2,xhor,yhor)
      call pgline(2,xvert,yvert)
      return
      end
c*********************************************
c      subroutine calobs3(cal,sobstot,im,jm,nm,nfb)
c      dimension sobstot(2000,200,24,10),cal(2000,200,24)
c      do n=1,nm
c         do j=1,jm  
c            do i=1,im
c               den=cal(i,j,n)
c               sobstot(i,j,n,nfb)=sobstot(i,j,n,nfb)/den   !  correction cal
c            enddo
c         enddo
c      enddo
c      return
c      end
c*************************************************
c*******************************
      

