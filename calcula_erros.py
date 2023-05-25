

# Criar as listas vazias para armazenar os conteúdos
reacoes_apoio = [3900.15177048, 2899.8185824, 1599.79834929]
deslocamentos = [0.0,0.0,0.00100455,-0.00431122,0.00035967,-0.00466257,0.00025962,-0.00443874,0.00055842,-0.00463706,-0.00032436,-0.00424731,0.00116427,0.0]
#deformacoes = [-0.00639476, -0.00449278, -0.00377605, -0.00352792, -0.00075546, 0.0007554, 0.00191115, -0.00191126, 0.00187329, 0.00103517, 0.00315547]
forcas_internas = [-6484.19147266, -4555.60757683, -3828.85492504, -3577.2578593, -766.02312311, 765.96588097, 1937.87876057,-1937.98588388, 1899.48539432, 1049.64239524, 3199.59669858]
tensoes_internas = [-1.23508409e+09, -8.67734777e+08, -7.29305700e+08, -6.81382449e+08, -1.45909166e+08, 1.45898263e+08, 3.69119764e+08, -3.69140168e+08, 3.61806742e+08, 1.99931885e+08, 6.09446990e+08]

reacoes_lisa =[3900.0, 2899.99999999999, 1600.0]
deslocamentos_lisa = [0, 0, 0.00100473773851939, -0.00431173336893045, 0.000359768635630702, -0.00466312908057704, 0.000259742558608514, -0.00443927764867052, 0.000558588144795037, -0.00463758827929192, -0.000324272590066568, -0.00424778496675707, 0.00116451426796254, 0]
forcas_internas_lisa = [-6484.59713474938, -4555.9885041558, -3829.26641146838, -3577.70876399966, -766.179646036105, 766.179646036095, 1937.9838105619, -1937.9838105619, 1899.99999999999, 1049.99999999999, 3199.99999999999]
tensoes_internas_lisa = [-1235161358.99988, -867807334.124915, -729384078.37493, -681468335.999936, -145938980.197353, 145938980.197351, 369139773.440362, -369139773.440363, 361904761.904759, 199999999.999998, 609523809.523808]

# Criar as listas vazias para armazenar os erros
erros_reacoes = []
erros_deslocamentos = []
erros_forcas_internas = []
erros_tensoes_internas = []

# Calcular os erros
for i in range(0, len(reacoes_apoio)):
    if reacoes_lisa[i] != 0:
        erros_reacoes.append(abs((reacoes_lisa[i] - reacoes_apoio[i]) / reacoes_lisa[i]))
    else:
        erros_reacoes.append(float(0))

for i in range(0, len(deslocamentos)):
    if deslocamentos_lisa[i] != 0:
        erros_deslocamentos.append(abs((deslocamentos_lisa[i] - deslocamentos[i]) / deslocamentos_lisa[i]))
    else:
        erros_deslocamentos.append(float(0))

for i in range(0, len(forcas_internas)):
    if forcas_internas_lisa[i] != 0:
        erros_forcas_internas.append(abs((forcas_internas_lisa[i] - forcas_internas[i]) / forcas_internas_lisa[i]))
    else:
        erros_forcas_internas.append(float(0))

for i in range(0, len(tensoes_internas)):
    if tensoes_internas_lisa[i] != 0:
        erros_tensoes_internas.append(abs((tensoes_internas_lisa[i] - tensoes_internas[i]) / tensoes_internas_lisa[i]))
    else:
        erros_tensoes_internas.append(float(0))
# Imprimir os erros
print('Erro relativo máximo das reações de apoio: {:.2e}%'.format(max(erros_reacoes) * 100))
print('Erro relativo máximo dos deslocamentos: {:.2e}%'.format(max(erros_deslocamentos) * 100))
print('Erro relativo máximo das forças internas: {:.2e}%'.format(max(erros_forcas_internas) * 100))
print('Erro relativo máximo das tensões internas: {:.2e}%'.format(max(erros_tensoes_internas) * 100))
print('O Erro relativo máximo das deformações não pode ser calculado pois não há informações sobre as deformações no LISA.')

# Agora sem o formato de porcentagem
print('\n')
print('E sem o formato de porcentagem:')
print('\nErro relativo máximo das reações de apoio: {:.2e}'.format(max(erros_reacoes)))
print('Erro relativo máximo dos deslocamentos: {:.2e}'.format(max(erros_deslocamentos)))
print('Erro relativo máximo das forças internas: {:.2e}'.format(max(erros_forcas_internas)))
print('Erro relativo máximo das tensões internas: {:.2e}'.format(max(erros_tensoes_internas)))
print('O Erro relativo máximo das deformações não pode ser calculado pois não há informações sobre as deformações no LISA.')