import cvxpy as cp
#import ecos

# 检查ECOS_BB是否可用
print("已安装求解器:", cp.installed_solvers())

# 测试混合整数问题
try:
    x = cp.Variable(integer=True)
    prob = cp.Problem(cp.Minimize(x), [x >= 1.5])
    prob.solve(solver=cp.ECOS_BB)
    print("ECOS_BB测试成功!")
    print("解:", x.value)
except Exception as e:
    print("ECOS_BB测试失败:", e)