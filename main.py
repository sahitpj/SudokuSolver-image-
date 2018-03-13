def check(k,small_grid,row,coloumn):
  if k not in list(small_grid[0]+small_grid[1]+small_grid[2]) and k not in row and k not in coloumn:
    return True
  else:
    return False

def find_col(grid,r,c):
  return [grid[i][c] for i in range(9)]

def solve(grid,r,c):
  a=(r//3)*3
  b=(c//3)*3
  sg=[grid[a][b:b+3],grid[a+1][b:b+3],grid[a+2][b:b+3]]
  
  if grid[r][c]!=0:
    if find_col(grid,r,c).count(grid[r][c])>1 or grid[r].count(grid[r][c])>1 or list(sg[0]+sg[1]+sg[2]).count(grid[r][c])>1:
      return False
    if c==8 and r<8:
      if solve(grid,r+1,0):
        return True
    elif c==8 and r==8:
      return True
    else:
      if solve(grid,r,c+1):
        return True
    return False
  
  col=find_col(grid,r,c)
  for i in range(1,10):
    if check(i,sg,grid[r],col):
      grid[r][c]=i
      if c==8 and r<8:
        if solve(grid,r+1,0):
          return True
      elif c==8 and r==8:
        return True
      else:
        if solve(grid,r,c+1):
          return True
  
  grid[r][c]=0
  return False

grid=[[int(i) for i in input().split()]for j in range(9)]
f=solve(grid,0,0)
if f==True:
  for i in range(9):
    print(*grid[i])
else:
  print("Invalid")