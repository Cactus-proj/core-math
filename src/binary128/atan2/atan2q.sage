# print K hard-to-round cases in file f using the algorithm described in
# "The CORE-MATH Project",
# by Alexei Sibidanov, Paul Zimmermann, Stéphane Glondu
# Proceedings of the 29th IEEE Symposium on Computer Arithmetic (ARITH 2022)
# end of Section IIB
# worst_cases("/tmp/out.wc",1000)
def worst_cases(f,K,min_m=113):
   maxm = 0
   f = open("/tmp/out"+str(min_m)+".wc","w")
   p = 113 # target precision
   R = RealField(p+1) # p+1 for having hard-to-round cases to nearest also
   while K>0:
      z = R.random_element()
      Z = z.exact_rational()
      t = n(tan(Z),1000)
      l = continued_fraction(t)
      for r in l.convergents():
         if r==0:
            continue
         if r.numer().nbits()>p or r.denom().nbits()>p:
            break
         y = R(r.numer())/2^p
         x = R(r.denom())/2^p
         m = identical_bits_atan2(y,x)
         if m>=min_m:
            f.write(get_hex(y)+" "+get_hex(x)+"\n")
            K -= 1
            break
   f.close()
