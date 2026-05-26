# print K hard-to-round cases in file f using the algorithm described in
# "The CORE-MATH Project",
# by Alexei Sibidanov, Paul Zimmermann, Stéphane Glondu
# Proceedings of the 29th IEEE Symposium on Computer Arithmetic (ARITH 2022)
# end of Section IIB
# worst_cases(1000)
def worst_cases(K,wanted_m=113):
   maxm = 0
   f = open("/tmp/out"+str(wanted_m)+".wc","w")
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
         if m==wanted_m:
            f.write(get_hex(y)+" "+get_hex(x)+"\n")
            K -= 1
            break
   f.close()

# hard-to-round cases in the subnormal range are those from division
# since atan(y/x) ~ y/x for tiny y/x
def worst_cases_subnormal(K,wanted_m=113):
   maxm = 0
   f = open("/tmp/out"+str(wanted_m)+".wc","w")
   R113 = RealField(113)
   while K>0:
      e = ZZ.random_element(-16493,-16381)
      p = 112 + e + 16382 # 1 <= p <= 112
      # in subnormal range [2^(e-1),2^e), numbers have p bits
      Y = ZZ.random_element(2^112,2^113) # Y has 113 bits
      # we want y/x near z thus y near x*z
      # where x has 113 bits and z has p or p+1 bits
      for t in [2^p*Y-1,2^p*Y+1,2^(p+1)*Y-1,2^(p+1)*Y+1]:
            for X in divisors(t):
               Z = t//X
               if X.nbits()<113 and Z.nbits()<=p+1:
                  x = R113(X/2^(e//2))
                  y = R113(Y*2^(e-(e//2)))
                  # X*Z ~ 2^p*Y or 2^(p+1)*Y
                  # thus Y/X ~ Z/2^p or Z/2^(p+1)
                  # thus y/x ~ Z*2^(e-p) or Z*2^(e-p-1)
                  # since Y >= 2^112 and X < 2^113, we have Y/X >= 1/2
                  # thus y/x = Y/X*2^e >= 2^(e-1)
                  while y/x>=2^e:
                     x = 2*x
                  f.write(get_hex(y)+" "+get_hex(x)+"\n")
                  f.flush()
                  K -= 1
                  print ("remains", K)
                  break
   f.close()
