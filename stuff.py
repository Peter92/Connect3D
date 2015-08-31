#short version of draw grid, S=grid_size, D=grid_data
R=range(S);k=0;L=[];G=S*2;F=G*2;H=G-1;a,b,c,d=list(' |/_')
for j in R:
 L+=[b+a*H+b+d*G+b+d*H+b if j else a*(G+1)+d*F]
 for i in R:
  Y=a*(G-i*2)+c+''.join((a+str(D[k+x]).ljust(1)+a+c)for x in R);k+=S;Z=a*(H-i*2)+c+'___/'*S
  if j!=R[-1]:Y+=a*(i*2)+b;Z+=a*(i*2+1)+b
  if j:Y,Z=[b+n[1:F+1]+b+n[F+2:]for n in Y,Z]
  L+=[Y,Z]
print '\n'.join(L)
