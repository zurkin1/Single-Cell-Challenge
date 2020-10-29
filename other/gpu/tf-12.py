import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations as cs
import sys
import time

num_situ = 20
batch_size = 500  # 100
two_list = [[[2 ** x] for x in range(0, num_situ)] for j in range(batch_size)]
glist = ['danr', 'CG14427', 'dan', 'CG43394', 'ImpL2', 'Nek2', 'CG8147', 'Ama', 'Btk29A', 'trn', 'numb', 'prd', 'brk',
         'tsh', 'pxb', 'dpn', 'ftz', 'Kr', 'h', 'eve', 'Traf4', 'run', 'Blimp-1', 'lok', 'kni', 'tkv', 'MESR3', 'odd',
         'noc', 'nub', 'Ilp4', 'aay', 'twi', 'bmm', 'hb', 'toc', 'rho', 'CG10479', 'gt', 'gk', 'apt', 'D', 'sna',
         'NetA', 'Mdr49', 'fj', 'Mes2', 'CG11208', 'Doc2', 'bun', 'tll', 'Cyp310a1', 'Doc3', 'htl', 'Esp', 'bowl', 'oc',
         'ImpE2', 'CG17724', 'fkh', 'edl', 'ems', 'zen2', 'CG17786', 'zen', 'disco', 'Dfd', 'mfas', 'knrl', 'Ance',
         'croc', 'rau', 'cnc', 'exex', 'cad', 'Antp', 'erm', 'ken', 'peb', 'srp', 'E(spl)m5-HLH', 'CenG1A', 'zfh1',
         'hkb']
bdtnp_bin = pd.read_csv('binarized_bdtnp.csv')
bdtnp = bdtnp_bin.values.astype(np.float32)


def generator():
    for tup in cs(glist, num_situ):  # sequence:
        yield bdtnp_bin[np.reshape(tup, -1)].values.astype(np.float32)


dataset = tf.data.Dataset().from_generator(generator,
                                           output_types=tf.float32,
                                           output_shapes=(tf.TensorShape([3039, num_situ]))).batch(batch_size).prefetch(
    4000000)  # 4000000
dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))  # tf.data.experimental.prefetch_to_device
iter = dataset.make_initializable_iterator()

with tf.device('/GPU:0'):
    #Y = tf.placeholder(tf.float32, shape=(batch_size, num_situ, 1))
    Y = tf.reshape(tf.tile(tf.constant([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288], tf.float32),
                           tf.constant([500])),
                   (500, 20, 1))
    X = iter.get_next()
    #total_max = tf.Variable(0)

    # Code binary sequences as integers in range [0,2**20].
    product = tf.matmul(X, Y)  # product.shape == (10, 3039, 1)
    # Move to vector
    un_2 = tf.to_int32(tf.squeeze(product))  # un_2.shape == (10, 3039)

    un_3_00, _ = tf.unique(un_2[0, :])
    un_3_01, _ = tf.unique(un_2[1, :])
    un_3_02, _ = tf.unique(un_2[2, :])
    un_3_03, _ = tf.unique(un_2[3, :])
    un_3_04, _ = tf.unique(un_2[4, :])
    un_3_05, _ = tf.unique(un_2[5, :])
    un_3_06, _ = tf.unique(un_2[6, :])
    un_3_07, _ = tf.unique(un_2[7, :])
    un_3_08, _ = tf.unique(un_2[8, :])
    un_3_09, _ = tf.unique(un_2[9, :])
    un_3_10, _ = tf.unique(un_2[10, :])
    un_3_11, _ = tf.unique(un_2[11, :])
    un_3_12, _ = tf.unique(un_2[12, :])
    un_3_13, _ = tf.unique(un_2[13, :])
    un_3_14, _ = tf.unique(un_2[14, :])
    un_3_15, _ = tf.unique(un_2[15, :])
    un_3_16, _ = tf.unique(un_2[16, :])
    un_3_17, _ = tf.unique(un_2[17, :])
    un_3_18, _ = tf.unique(un_2[18, :])
    un_3_19, _ = tf.unique(un_2[19, :])
    un_3_20, _ = tf.unique(un_2[20, :])
    un_3_21, _ = tf.unique(un_2[21, :])
    un_3_22, _ = tf.unique(un_2[22, :])
    un_3_23, _ = tf.unique(un_2[23, :])
    un_3_24, _ = tf.unique(un_2[24, :])
    un_3_25, _ = tf.unique(un_2[25, :])
    un_3_26, _ = tf.unique(un_2[26, :])
    un_3_27, _ = tf.unique(un_2[27, :])
    un_3_28, _ = tf.unique(un_2[28, :])
    un_3_29, _ = tf.unique(un_2[29, :])
    un_3_30, _ = tf.unique(un_2[30, :])
    un_3_31, _ = tf.unique(un_2[31, :])
    un_3_32, _ = tf.unique(un_2[32, :])
    un_3_33, _ = tf.unique(un_2[33, :])
    un_3_34, _ = tf.unique(un_2[34, :])
    un_3_35, _ = tf.unique(un_2[35, :])
    un_3_36, _ = tf.unique(un_2[36, :])
    un_3_37, _ = tf.unique(un_2[37, :])
    un_3_38, _ = tf.unique(un_2[38, :])
    un_3_39, _ = tf.unique(un_2[39, :])
    un_3_40, _ = tf.unique(un_2[40, :])
    un_3_41, _ = tf.unique(un_2[41, :])
    un_3_42, _ = tf.unique(un_2[42, :])
    un_3_43, _ = tf.unique(un_2[43, :])
    un_3_44, _ = tf.unique(un_2[44, :])
    un_3_45, _ = tf.unique(un_2[45, :])
    un_3_46, _ = tf.unique(un_2[46, :])
    un_3_47, _ = tf.unique(un_2[47, :])
    un_3_48, _ = tf.unique(un_2[48, :])
    un_3_49, _ = tf.unique(un_2[49, :])
    un_3_50, _ = tf.unique(un_2[50, :])
    un_3_51, _ = tf.unique(un_2[51, :])
    un_3_52, _ = tf.unique(un_2[52, :])
    un_3_53, _ = tf.unique(un_2[53, :])
    un_3_54, _ = tf.unique(un_2[54, :])
    un_3_55, _ = tf.unique(un_2[55, :])
    un_3_56, _ = tf.unique(un_2[56, :])
    un_3_57, _ = tf.unique(un_2[57, :])
    un_3_58, _ = tf.unique(un_2[58, :])
    un_3_59, _ = tf.unique(un_2[59, :])
    un_3_60, _ = tf.unique(un_2[60, :])
    un_3_61, _ = tf.unique(un_2[61, :])
    un_3_62, _ = tf.unique(un_2[62, :])
    un_3_63, _ = tf.unique(un_2[63, :])
    un_3_64, _ = tf.unique(un_2[64, :])
    un_3_65, _ = tf.unique(un_2[65, :])
    un_3_66, _ = tf.unique(un_2[66, :])
    un_3_67, _ = tf.unique(un_2[67, :])
    un_3_68, _ = tf.unique(un_2[68, :])
    un_3_69, _ = tf.unique(un_2[69, :])
    un_3_70, _ = tf.unique(un_2[70, :])
    un_3_71, _ = tf.unique(un_2[71, :])
    un_3_72, _ = tf.unique(un_2[72, :])
    un_3_73, _ = tf.unique(un_2[73, :])
    un_3_74, _ = tf.unique(un_2[74, :])
    un_3_75, _ = tf.unique(un_2[75, :])
    un_3_76, _ = tf.unique(un_2[76, :])
    un_3_77, _ = tf.unique(un_2[77, :])
    un_3_78, _ = tf.unique(un_2[78, :])
    un_3_79, _ = tf.unique(un_2[79, :])
    un_3_80, _ = tf.unique(un_2[80, :])
    un_3_81, _ = tf.unique(un_2[81, :])
    un_3_82, _ = tf.unique(un_2[82, :])
    un_3_83, _ = tf.unique(un_2[83, :])
    un_3_84, _ = tf.unique(un_2[84, :])
    un_3_85, _ = tf.unique(un_2[85, :])
    un_3_86, _ = tf.unique(un_2[86, :])
    un_3_87, _ = tf.unique(un_2[87, :])
    un_3_88, _ = tf.unique(un_2[88, :])
    un_3_89, _ = tf.unique(un_2[89, :])
    un_3_90, _ = tf.unique(un_2[90, :])
    un_3_91, _ = tf.unique(un_2[91, :])
    un_3_92, _ = tf.unique(un_2[92, :])
    un_3_93, _ = tf.unique(un_2[93, :])
    un_3_94, _ = tf.unique(un_2[94, :])
    un_3_95, _ = tf.unique(un_2[95, :])
    un_3_96, _ = tf.unique(un_2[96, :])
    un_3_97, _ = tf.unique(un_2[97, :])
    un_3_98, _ = tf.unique(un_2[98, :])
    un_3_99, _ = tf.unique(un_2[99, :])
    un_3_100, _ = tf.unique(un_2[100, :])
    un_3_101, _ = tf.unique(un_2[101, :])
    un_3_102, _ = tf.unique(un_2[102, :])
    un_3_103, _ = tf.unique(un_2[103, :])
    un_3_104, _ = tf.unique(un_2[104, :])
    un_3_105, _ = tf.unique(un_2[105, :])
    un_3_106, _ = tf.unique(un_2[106, :])
    un_3_107, _ = tf.unique(un_2[107, :])
    un_3_108, _ = tf.unique(un_2[108, :])
    un_3_109, _ = tf.unique(un_2[109, :])
    un_3_110, _ = tf.unique(un_2[110, :])
    un_3_111, _ = tf.unique(un_2[111, :])
    un_3_112, _ = tf.unique(un_2[112, :])
    un_3_113, _ = tf.unique(un_2[113, :])
    un_3_114, _ = tf.unique(un_2[114, :])
    un_3_115, _ = tf.unique(un_2[115, :])
    un_3_116, _ = tf.unique(un_2[116, :])
    un_3_117, _ = tf.unique(un_2[117, :])
    un_3_118, _ = tf.unique(un_2[118, :])
    un_3_119, _ = tf.unique(un_2[119, :])
    un_3_120, _ = tf.unique(un_2[120, :])
    un_3_121, _ = tf.unique(un_2[121, :])
    un_3_122, _ = tf.unique(un_2[122, :])
    un_3_123, _ = tf.unique(un_2[123, :])
    un_3_124, _ = tf.unique(un_2[124, :])
    un_3_125, _ = tf.unique(un_2[125, :])
    un_3_126, _ = tf.unique(un_2[126, :])
    un_3_127, _ = tf.unique(un_2[127, :])
    un_3_128, _ = tf.unique(un_2[128, :])
    un_3_129, _ = tf.unique(un_2[129, :])
    un_3_130, _ = tf.unique(un_2[130, :])
    un_3_131, _ = tf.unique(un_2[131, :])
    un_3_132, _ = tf.unique(un_2[132, :])
    un_3_133, _ = tf.unique(un_2[133, :])
    un_3_134, _ = tf.unique(un_2[134, :])
    un_3_135, _ = tf.unique(un_2[135, :])
    un_3_136, _ = tf.unique(un_2[136, :])
    un_3_137, _ = tf.unique(un_2[137, :])
    un_3_138, _ = tf.unique(un_2[138, :])
    un_3_139, _ = tf.unique(un_2[139, :])
    un_3_140, _ = tf.unique(un_2[140, :])
    un_3_141, _ = tf.unique(un_2[141, :])
    un_3_142, _ = tf.unique(un_2[142, :])
    un_3_143, _ = tf.unique(un_2[143, :])
    un_3_144, _ = tf.unique(un_2[144, :])
    un_3_145, _ = tf.unique(un_2[145, :])
    un_3_146, _ = tf.unique(un_2[146, :])
    un_3_147, _ = tf.unique(un_2[147, :])
    un_3_148, _ = tf.unique(un_2[148, :])
    un_3_149, _ = tf.unique(un_2[149, :])
    un_3_150, _ = tf.unique(un_2[150, :])
    un_3_151, _ = tf.unique(un_2[151, :])
    un_3_152, _ = tf.unique(un_2[152, :])
    un_3_153, _ = tf.unique(un_2[153, :])
    un_3_154, _ = tf.unique(un_2[154, :])
    un_3_155, _ = tf.unique(un_2[155, :])
    un_3_156, _ = tf.unique(un_2[156, :])
    un_3_157, _ = tf.unique(un_2[157, :])
    un_3_158, _ = tf.unique(un_2[158, :])
    un_3_159, _ = tf.unique(un_2[159, :])
    un_3_160, _ = tf.unique(un_2[160, :])
    un_3_161, _ = tf.unique(un_2[161, :])
    un_3_162, _ = tf.unique(un_2[162, :])
    un_3_163, _ = tf.unique(un_2[163, :])
    un_3_164, _ = tf.unique(un_2[164, :])
    un_3_165, _ = tf.unique(un_2[165, :])
    un_3_166, _ = tf.unique(un_2[166, :])
    un_3_167, _ = tf.unique(un_2[167, :])
    un_3_168, _ = tf.unique(un_2[168, :])
    un_3_169, _ = tf.unique(un_2[169, :])
    un_3_170, _ = tf.unique(un_2[170, :])
    un_3_171, _ = tf.unique(un_2[171, :])
    un_3_172, _ = tf.unique(un_2[172, :])
    un_3_173, _ = tf.unique(un_2[173, :])
    un_3_174, _ = tf.unique(un_2[174, :])
    un_3_175, _ = tf.unique(un_2[175, :])
    un_3_176, _ = tf.unique(un_2[176, :])
    un_3_177, _ = tf.unique(un_2[177, :])
    un_3_178, _ = tf.unique(un_2[178, :])
    un_3_179, _ = tf.unique(un_2[179, :])
    un_3_180, _ = tf.unique(un_2[180, :])
    un_3_181, _ = tf.unique(un_2[181, :])
    un_3_182, _ = tf.unique(un_2[182, :])
    un_3_183, _ = tf.unique(un_2[183, :])
    un_3_184, _ = tf.unique(un_2[184, :])
    un_3_185, _ = tf.unique(un_2[185, :])
    un_3_186, _ = tf.unique(un_2[186, :])
    un_3_187, _ = tf.unique(un_2[187, :])
    un_3_188, _ = tf.unique(un_2[188, :])
    un_3_189, _ = tf.unique(un_2[189, :])
    un_3_190, _ = tf.unique(un_2[190, :])
    un_3_191, _ = tf.unique(un_2[191, :])
    un_3_192, _ = tf.unique(un_2[192, :])
    un_3_193, _ = tf.unique(un_2[193, :])
    un_3_194, _ = tf.unique(un_2[194, :])
    un_3_195, _ = tf.unique(un_2[195, :])
    un_3_196, _ = tf.unique(un_2[196, :])
    un_3_197, _ = tf.unique(un_2[197, :])
    un_3_198, _ = tf.unique(un_2[198, :])
    un_3_199, _ = tf.unique(un_2[199, :])
    un_3_200, _ = tf.unique(un_2[200, :])
    un_3_201, _ = tf.unique(un_2[201, :])
    un_3_202, _ = tf.unique(un_2[202, :])
    un_3_203, _ = tf.unique(un_2[203, :])
    un_3_204, _ = tf.unique(un_2[204, :])
    un_3_205, _ = tf.unique(un_2[205, :])
    un_3_206, _ = tf.unique(un_2[206, :])
    un_3_207, _ = tf.unique(un_2[207, :])
    un_3_208, _ = tf.unique(un_2[208, :])
    un_3_209, _ = tf.unique(un_2[209, :])
    un_3_210, _ = tf.unique(un_2[210, :])
    un_3_211, _ = tf.unique(un_2[211, :])
    un_3_212, _ = tf.unique(un_2[212, :])
    un_3_213, _ = tf.unique(un_2[213, :])
    un_3_214, _ = tf.unique(un_2[214, :])
    un_3_215, _ = tf.unique(un_2[215, :])
    un_3_216, _ = tf.unique(un_2[216, :])
    un_3_217, _ = tf.unique(un_2[217, :])
    un_3_218, _ = tf.unique(un_2[218, :])
    un_3_219, _ = tf.unique(un_2[219, :])
    un_3_220, _ = tf.unique(un_2[220, :])
    un_3_221, _ = tf.unique(un_2[221, :])
    un_3_222, _ = tf.unique(un_2[222, :])
    un_3_223, _ = tf.unique(un_2[223, :])
    un_3_224, _ = tf.unique(un_2[224, :])
    un_3_225, _ = tf.unique(un_2[225, :])
    un_3_226, _ = tf.unique(un_2[226, :])
    un_3_227, _ = tf.unique(un_2[227, :])
    un_3_228, _ = tf.unique(un_2[228, :])
    un_3_229, _ = tf.unique(un_2[229, :])
    un_3_230, _ = tf.unique(un_2[230, :])
    un_3_231, _ = tf.unique(un_2[231, :])
    un_3_232, _ = tf.unique(un_2[232, :])
    un_3_233, _ = tf.unique(un_2[233, :])
    un_3_234, _ = tf.unique(un_2[234, :])
    un_3_235, _ = tf.unique(un_2[235, :])
    un_3_236, _ = tf.unique(un_2[236, :])
    un_3_237, _ = tf.unique(un_2[237, :])
    un_3_238, _ = tf.unique(un_2[238, :])
    un_3_239, _ = tf.unique(un_2[239, :])
    un_3_240, _ = tf.unique(un_2[240, :])
    un_3_241, _ = tf.unique(un_2[241, :])
    un_3_242, _ = tf.unique(un_2[242, :])
    un_3_243, _ = tf.unique(un_2[243, :])
    un_3_244, _ = tf.unique(un_2[244, :])
    un_3_245, _ = tf.unique(un_2[245, :])
    un_3_246, _ = tf.unique(un_2[246, :])
    un_3_247, _ = tf.unique(un_2[247, :])
    un_3_248, _ = tf.unique(un_2[248, :])
    un_3_249, _ = tf.unique(un_2[249, :])
    un_3_250, _ = tf.unique(un_2[250, :])
    un_3_251, _ = tf.unique(un_2[251, :])
    un_3_252, _ = tf.unique(un_2[252, :])
    un_3_253, _ = tf.unique(un_2[253, :])
    un_3_254, _ = tf.unique(un_2[254, :])
    un_3_255, _ = tf.unique(un_2[255, :])
    un_3_256, _ = tf.unique(un_2[256, :])
    un_3_257, _ = tf.unique(un_2[257, :])
    un_3_258, _ = tf.unique(un_2[258, :])
    un_3_259, _ = tf.unique(un_2[259, :])
    un_3_260, _ = tf.unique(un_2[260, :])
    un_3_261, _ = tf.unique(un_2[261, :])
    un_3_262, _ = tf.unique(un_2[262, :])
    un_3_263, _ = tf.unique(un_2[263, :])
    un_3_264, _ = tf.unique(un_2[264, :])
    un_3_265, _ = tf.unique(un_2[265, :])
    un_3_266, _ = tf.unique(un_2[266, :])
    un_3_267, _ = tf.unique(un_2[267, :])
    un_3_268, _ = tf.unique(un_2[268, :])
    un_3_269, _ = tf.unique(un_2[269, :])
    un_3_270, _ = tf.unique(un_2[270, :])
    un_3_271, _ = tf.unique(un_2[271, :])
    un_3_272, _ = tf.unique(un_2[272, :])
    un_3_273, _ = tf.unique(un_2[273, :])
    un_3_274, _ = tf.unique(un_2[274, :])
    un_3_275, _ = tf.unique(un_2[275, :])
    un_3_276, _ = tf.unique(un_2[276, :])
    un_3_277, _ = tf.unique(un_2[277, :])
    un_3_278, _ = tf.unique(un_2[278, :])
    un_3_279, _ = tf.unique(un_2[279, :])
    un_3_280, _ = tf.unique(un_2[280, :])
    un_3_281, _ = tf.unique(un_2[281, :])
    un_3_282, _ = tf.unique(un_2[282, :])
    un_3_283, _ = tf.unique(un_2[283, :])
    un_3_284, _ = tf.unique(un_2[284, :])
    un_3_285, _ = tf.unique(un_2[285, :])
    un_3_286, _ = tf.unique(un_2[286, :])
    un_3_287, _ = tf.unique(un_2[287, :])
    un_3_288, _ = tf.unique(un_2[288, :])
    un_3_289, _ = tf.unique(un_2[289, :])
    un_3_290, _ = tf.unique(un_2[290, :])
    un_3_291, _ = tf.unique(un_2[291, :])
    un_3_292, _ = tf.unique(un_2[292, :])
    un_3_293, _ = tf.unique(un_2[293, :])
    un_3_294, _ = tf.unique(un_2[294, :])
    un_3_295, _ = tf.unique(un_2[295, :])
    un_3_296, _ = tf.unique(un_2[296, :])
    un_3_297, _ = tf.unique(un_2[297, :])
    un_3_298, _ = tf.unique(un_2[298, :])
    un_3_299, _ = tf.unique(un_2[299, :])
    un_3_300, _ = tf.unique(un_2[300,:])
    un_3_301, _ = tf.unique(un_2[301,:])
    un_3_302, _ = tf.unique(un_2[302,:])
    un_3_303, _ = tf.unique(un_2[303,:])
    un_3_304, _ = tf.unique(un_2[304,:])
    un_3_305, _ = tf.unique(un_2[305,:])
    un_3_306, _ = tf.unique(un_2[306,:])
    un_3_307, _ = tf.unique(un_2[307,:])
    un_3_308, _ = tf.unique(un_2[308,:])
    un_3_309, _ = tf.unique(un_2[309,:])
    un_3_310, _ = tf.unique(un_2[310,:])
    un_3_311, _ = tf.unique(un_2[311,:])
    un_3_312, _ = tf.unique(un_2[312,:])
    un_3_313, _ = tf.unique(un_2[313,:])
    un_3_314, _ = tf.unique(un_2[314,:])
    un_3_315, _ = tf.unique(un_2[315,:])
    un_3_316, _ = tf.unique(un_2[316,:])
    un_3_317, _ = tf.unique(un_2[317,:])
    un_3_318, _ = tf.unique(un_2[318,:])
    un_3_319, _ = tf.unique(un_2[319,:])
    un_3_320, _ = tf.unique(un_2[320,:])
    un_3_321, _ = tf.unique(un_2[321,:])
    un_3_322, _ = tf.unique(un_2[322,:])
    un_3_323, _ = tf.unique(un_2[323,:])
    un_3_324, _ = tf.unique(un_2[324,:])
    un_3_325, _ = tf.unique(un_2[325,:])
    un_3_326, _ = tf.unique(un_2[326,:])
    un_3_327, _ = tf.unique(un_2[327,:])
    un_3_328, _ = tf.unique(un_2[328,:])
    un_3_329, _ = tf.unique(un_2[329,:])
    un_3_330, _ = tf.unique(un_2[330,:])
    un_3_331, _ = tf.unique(un_2[331,:])
    un_3_332, _ = tf.unique(un_2[332,:])
    un_3_333, _ = tf.unique(un_2[333,:])
    un_3_334, _ = tf.unique(un_2[334,:])
    un_3_335, _ = tf.unique(un_2[335,:])
    un_3_336, _ = tf.unique(un_2[336,:])
    un_3_337, _ = tf.unique(un_2[337,:])
    un_3_338, _ = tf.unique(un_2[338,:])
    un_3_339, _ = tf.unique(un_2[339,:])
    un_3_340, _ = tf.unique(un_2[340,:])
    un_3_341, _ = tf.unique(un_2[341,:])
    un_3_342, _ = tf.unique(un_2[342,:])
    un_3_343, _ = tf.unique(un_2[343,:])
    un_3_344, _ = tf.unique(un_2[344,:])
    un_3_345, _ = tf.unique(un_2[345,:])
    un_3_346, _ = tf.unique(un_2[346,:])
    un_3_347, _ = tf.unique(un_2[347,:])
    un_3_348, _ = tf.unique(un_2[348,:])
    un_3_349, _ = tf.unique(un_2[349,:])
    un_3_350, _ = tf.unique(un_2[350,:])
    un_3_351, _ = tf.unique(un_2[351,:])
    un_3_352, _ = tf.unique(un_2[352,:])
    un_3_353, _ = tf.unique(un_2[353,:])
    un_3_354, _ = tf.unique(un_2[354,:])
    un_3_355, _ = tf.unique(un_2[355,:])
    un_3_356, _ = tf.unique(un_2[356,:])
    un_3_357, _ = tf.unique(un_2[357,:])
    un_3_358, _ = tf.unique(un_2[358,:])
    un_3_359, _ = tf.unique(un_2[359,:])
    un_3_360, _ = tf.unique(un_2[360,:])
    un_3_361, _ = tf.unique(un_2[361,:])
    un_3_362, _ = tf.unique(un_2[362,:])
    un_3_363, _ = tf.unique(un_2[363,:])
    un_3_364, _ = tf.unique(un_2[364,:])
    un_3_365, _ = tf.unique(un_2[365,:])
    un_3_366, _ = tf.unique(un_2[366,:])
    un_3_367, _ = tf.unique(un_2[367,:])
    un_3_368, _ = tf.unique(un_2[368,:])
    un_3_369, _ = tf.unique(un_2[369,:])
    un_3_370, _ = tf.unique(un_2[370,:])
    un_3_371, _ = tf.unique(un_2[371,:])
    un_3_372, _ = tf.unique(un_2[372,:])
    un_3_373, _ = tf.unique(un_2[373,:])
    un_3_374, _ = tf.unique(un_2[374,:])
    un_3_375, _ = tf.unique(un_2[375,:])
    un_3_376, _ = tf.unique(un_2[376,:])
    un_3_377, _ = tf.unique(un_2[377,:])
    un_3_378, _ = tf.unique(un_2[378,:])
    un_3_379, _ = tf.unique(un_2[379,:])
    un_3_380, _ = tf.unique(un_2[380,:])
    un_3_381, _ = tf.unique(un_2[381,:])
    un_3_382, _ = tf.unique(un_2[382,:])
    un_3_383, _ = tf.unique(un_2[383,:])
    un_3_384, _ = tf.unique(un_2[384,:])
    un_3_385, _ = tf.unique(un_2[385,:])
    un_3_386, _ = tf.unique(un_2[386,:])
    un_3_387, _ = tf.unique(un_2[387,:])
    un_3_388, _ = tf.unique(un_2[388,:])
    un_3_389, _ = tf.unique(un_2[389,:])
    un_3_390, _ = tf.unique(un_2[390,:])
    un_3_391, _ = tf.unique(un_2[391,:])
    un_3_392, _ = tf.unique(un_2[392,:])
    un_3_393, _ = tf.unique(un_2[393,:])
    un_3_394, _ = tf.unique(un_2[394,:])
    un_3_395, _ = tf.unique(un_2[395,:])
    un_3_396, _ = tf.unique(un_2[396,:])
    un_3_397, _ = tf.unique(un_2[397,:])
    un_3_398, _ = tf.unique(un_2[398,:])
    un_3_399, _ = tf.unique(un_2[399,:])
    un_3_400, _ = tf.unique(un_2[400,:])
    un_3_401, _ = tf.unique(un_2[401,:])
    un_3_402, _ = tf.unique(un_2[402,:])
    un_3_403, _ = tf.unique(un_2[403,:])
    un_3_404, _ = tf.unique(un_2[404,:])
    un_3_405, _ = tf.unique(un_2[405,:])
    un_3_406, _ = tf.unique(un_2[406,:])
    un_3_407, _ = tf.unique(un_2[407,:])
    un_3_408, _ = tf.unique(un_2[408,:])
    un_3_409, _ = tf.unique(un_2[409,:])
    un_3_410, _ = tf.unique(un_2[410,:])
    un_3_411, _ = tf.unique(un_2[411,:])
    un_3_412, _ = tf.unique(un_2[412,:])
    un_3_413, _ = tf.unique(un_2[413,:])
    un_3_414, _ = tf.unique(un_2[414,:])
    un_3_415, _ = tf.unique(un_2[415,:])
    un_3_416, _ = tf.unique(un_2[416,:])
    un_3_417, _ = tf.unique(un_2[417,:])
    un_3_418, _ = tf.unique(un_2[418,:])
    un_3_419, _ = tf.unique(un_2[419,:])
    un_3_420, _ = tf.unique(un_2[420,:])
    un_3_421, _ = tf.unique(un_2[421,:])
    un_3_422, _ = tf.unique(un_2[422,:])
    un_3_423, _ = tf.unique(un_2[423,:])
    un_3_424, _ = tf.unique(un_2[424,:])
    un_3_425, _ = tf.unique(un_2[425,:])
    un_3_426, _ = tf.unique(un_2[426,:])
    un_3_427, _ = tf.unique(un_2[427,:])
    un_3_428, _ = tf.unique(un_2[428,:])
    un_3_429, _ = tf.unique(un_2[429,:])
    un_3_430, _ = tf.unique(un_2[430,:])
    un_3_431, _ = tf.unique(un_2[431,:])
    un_3_432, _ = tf.unique(un_2[432,:])
    un_3_433, _ = tf.unique(un_2[433,:])
    un_3_434, _ = tf.unique(un_2[434,:])
    un_3_435, _ = tf.unique(un_2[435,:])
    un_3_436, _ = tf.unique(un_2[436,:])
    un_3_437, _ = tf.unique(un_2[437,:])
    un_3_438, _ = tf.unique(un_2[438,:])
    un_3_439, _ = tf.unique(un_2[439,:])
    un_3_440, _ = tf.unique(un_2[440,:])
    un_3_441, _ = tf.unique(un_2[441,:])
    un_3_442, _ = tf.unique(un_2[442,:])
    un_3_443, _ = tf.unique(un_2[443,:])
    un_3_444, _ = tf.unique(un_2[444,:])
    un_3_445, _ = tf.unique(un_2[445,:])
    un_3_446, _ = tf.unique(un_2[446,:])
    un_3_447, _ = tf.unique(un_2[447,:])
    un_3_448, _ = tf.unique(un_2[448,:])
    un_3_449, _ = tf.unique(un_2[449,:])
    un_3_450, _ = tf.unique(un_2[450,:])
    un_3_451, _ = tf.unique(un_2[451,:])
    un_3_452, _ = tf.unique(un_2[452,:])
    un_3_453, _ = tf.unique(un_2[453,:])
    un_3_454, _ = tf.unique(un_2[454,:])
    un_3_455, _ = tf.unique(un_2[455,:])
    un_3_456, _ = tf.unique(un_2[456,:])
    un_3_457, _ = tf.unique(un_2[457,:])
    un_3_458, _ = tf.unique(un_2[458,:])
    un_3_459, _ = tf.unique(un_2[459,:])
    un_3_460, _ = tf.unique(un_2[460,:])
    un_3_461, _ = tf.unique(un_2[461,:])
    un_3_462, _ = tf.unique(un_2[462,:])
    un_3_463, _ = tf.unique(un_2[463,:])
    un_3_464, _ = tf.unique(un_2[464,:])
    un_3_465, _ = tf.unique(un_2[465,:])
    un_3_466, _ = tf.unique(un_2[466,:])
    un_3_467, _ = tf.unique(un_2[467,:])
    un_3_468, _ = tf.unique(un_2[468,:])
    un_3_469, _ = tf.unique(un_2[469,:])
    un_3_470, _ = tf.unique(un_2[470,:])
    un_3_471, _ = tf.unique(un_2[471,:])
    un_3_472, _ = tf.unique(un_2[472,:])
    un_3_473, _ = tf.unique(un_2[473,:])
    un_3_474, _ = tf.unique(un_2[474,:])
    un_3_475, _ = tf.unique(un_2[475,:])
    un_3_476, _ = tf.unique(un_2[476,:])
    un_3_477, _ = tf.unique(un_2[477,:])
    un_3_478, _ = tf.unique(un_2[478,:])
    un_3_479, _ = tf.unique(un_2[479,:])
    un_3_480, _ = tf.unique(un_2[480,:])
    un_3_481, _ = tf.unique(un_2[481,:])
    un_3_482, _ = tf.unique(un_2[482,:])
    un_3_483, _ = tf.unique(un_2[483,:])
    un_3_484, _ = tf.unique(un_2[484,:])
    un_3_485, _ = tf.unique(un_2[485,:])
    un_3_486, _ = tf.unique(un_2[486,:])
    un_3_487, _ = tf.unique(un_2[487,:])
    un_3_488, _ = tf.unique(un_2[488,:])
    un_3_489, _ = tf.unique(un_2[489,:])
    un_3_490, _ = tf.unique(un_2[490,:])
    un_3_491, _ = tf.unique(un_2[491,:])
    un_3_492, _ = tf.unique(un_2[492,:])
    un_3_493, _ = tf.unique(un_2[493,:])
    un_3_494, _ = tf.unique(un_2[494,:])
    un_3_495, _ = tf.unique(un_2[495,:])
    un_3_496, _ = tf.unique(un_2[496,:])
    un_3_497, _ = tf.unique(un_2[497,:])
    un_3_498, _ = tf.unique(un_2[498,:])
    un_3_499, _ = tf.unique(un_2[499,:])

    #all_max = tf.reduce_max(tf.stack(
    #    [all_max_0, all_max_1, all_max_2, all_max_3, all_max_4, all_max_5, all_max_6, all_max_7, all_max_8, all_max_9, \
    #     all_max_10, all_max_11, all_max_12, all_max_13, all_max_14, all_max_15, all_max_16, all_max_17, all_max_18,all_max_19, \
    #     all_max_20, all_max_21, all_max_22, all_max_23, all_max_24, all_max_25, all_max_26, all_max_27, all_max_28,all_max_29]))
    #max_op = tf.cond(all_max > total_max, total_max.assign(all_max))
    all_max_0 = tf.reduce_max(\
    tf.stack([tf.size(un_3_00), tf.size(un_3_01), tf.size(un_3_02), tf.size(un_3_03), tf.size(un_3_04), tf.size(un_3_05), tf.size(un_3_06), tf.size(un_3_07), tf.size(un_3_08), tf.size(un_3_09),\
    tf.size(un_3_10), tf.size(un_3_11), tf.size(un_3_12), tf.size(un_3_13), tf.size(un_3_14), tf.size(un_3_15), tf.size(un_3_16), tf.size(un_3_17), tf.size(un_3_18), tf.size(un_3_19),\
    tf.size(un_3_20), tf.size(un_3_21), tf.size(un_3_22), tf.size(un_3_23), tf.size(un_3_24), tf.size(un_3_25), tf.size(un_3_26), tf.size(un_3_27), tf.size(un_3_28), tf.size(un_3_29),\
    tf.size(un_3_30), tf.size(un_3_31), tf.size(un_3_32), tf.size(un_3_33), tf.size(un_3_34), tf.size(un_3_35), tf.size(un_3_36), tf.size(un_3_37), tf.size(un_3_38), tf.size(un_3_39),\
    tf.size(un_3_40), tf.size(un_3_41), tf.size(un_3_42), tf.size(un_3_43), tf.size(un_3_44), tf.size(un_3_45), tf.size(un_3_46), tf.size(un_3_47), tf.size(un_3_48), tf.size(un_3_49),\
    tf.size(un_3_50), tf.size(un_3_51), tf.size(un_3_52), tf.size(un_3_53), tf.size(un_3_54), tf.size(un_3_55), tf.size(un_3_56), tf.size(un_3_57), tf.size(un_3_58), tf.size(un_3_59),\
    tf.size(un_3_60), tf.size(un_3_61), tf.size(un_3_62), tf.size(un_3_63), tf.size(un_3_64), tf.size(un_3_65), tf.size(un_3_66), tf.size(un_3_67), tf.size(un_3_68), tf.size(un_3_69),\
    tf.size(un_3_70), tf.size(un_3_71), tf.size(un_3_72), tf.size(un_3_73), tf.size(un_3_74), tf.size(un_3_75), tf.size(un_3_76), tf.size(un_3_77), tf.size(un_3_78), tf.size(un_3_79),\
    tf.size(un_3_80), tf.size(un_3_81), tf.size(un_3_82), tf.size(un_3_83), tf.size(un_3_84), tf.size(un_3_85), tf.size(un_3_86), tf.size(un_3_87), tf.size(un_3_88), tf.size(un_3_89),\
    tf.size(un_3_90), tf.size(un_3_91), tf.size(un_3_92), tf.size(un_3_93), tf.size(un_3_94), tf.size(un_3_95), tf.size(un_3_96), tf.size(un_3_97), tf.size(un_3_98), tf.size(un_3_99),\
    tf.size(un_3_100), tf.size(un_3_101), tf.size(un_3_102), tf.size(un_3_103), tf.size(un_3_104), tf.size(un_3_105), tf.size(un_3_106), tf.size(un_3_107), tf.size(un_3_108), tf.size(un_3_109),\
    tf.size(un_3_110), tf.size(un_3_111), tf.size(un_3_112), tf.size(un_3_113), tf.size(un_3_114), tf.size(un_3_115), tf.size(un_3_116), tf.size(un_3_117), tf.size(un_3_118), tf.size(un_3_119),\
    tf.size(un_3_120), tf.size(un_3_121), tf.size(un_3_122), tf.size(un_3_123), tf.size(un_3_124), tf.size(un_3_125), tf.size(un_3_126), tf.size(un_3_127), tf.size(un_3_128), tf.size(un_3_129),\
    tf.size(un_3_130), tf.size(un_3_131), tf.size(un_3_132), tf.size(un_3_133), tf.size(un_3_134), tf.size(un_3_135), tf.size(un_3_136), tf.size(un_3_137), tf.size(un_3_138), tf.size(un_3_139),\
    tf.size(un_3_140), tf.size(un_3_141), tf.size(un_3_142), tf.size(un_3_143), tf.size(un_3_144), tf.size(un_3_145), tf.size(un_3_146), tf.size(un_3_147), tf.size(un_3_148), tf.size(un_3_149),\
    tf.size(un_3_150), tf.size(un_3_151), tf.size(un_3_152), tf.size(un_3_153), tf.size(un_3_154), tf.size(un_3_155), tf.size(un_3_156), tf.size(un_3_157), tf.size(un_3_158), tf.size(un_3_159),\
    tf.size(un_3_160), tf.size(un_3_161), tf.size(un_3_162), tf.size(un_3_163), tf.size(un_3_164), tf.size(un_3_165), tf.size(un_3_166), tf.size(un_3_167), tf.size(un_3_168), tf.size(un_3_169),\
    tf.size(un_3_170), tf.size(un_3_171), tf.size(un_3_172), tf.size(un_3_173), tf.size(un_3_174), tf.size(un_3_175), tf.size(un_3_176), tf.size(un_3_177), tf.size(un_3_178), tf.size(un_3_179),\
    tf.size(un_3_180), tf.size(un_3_181), tf.size(un_3_182), tf.size(un_3_183), tf.size(un_3_184), tf.size(un_3_185), tf.size(un_3_186), tf.size(un_3_187), tf.size(un_3_188), tf.size(un_3_189),\
    tf.size(un_3_190), tf.size(un_3_191), tf.size(un_3_192), tf.size(un_3_193), tf.size(un_3_194), tf.size(un_3_195), tf.size(un_3_196), tf.size(un_3_197), tf.size(un_3_198), tf.size(un_3_199),\
    tf.size(un_3_200), tf.size(un_3_201), tf.size(un_3_202), tf.size(un_3_203), tf.size(un_3_204), tf.size(un_3_205), tf.size(un_3_206), tf.size(un_3_207), tf.size(un_3_208), tf.size(un_3_209),\
    tf.size(un_3_210), tf.size(un_3_211), tf.size(un_3_212), tf.size(un_3_213), tf.size(un_3_214), tf.size(un_3_215), tf.size(un_3_216), tf.size(un_3_217), tf.size(un_3_218), tf.size(un_3_219),\
    tf.size(un_3_220), tf.size(un_3_221), tf.size(un_3_222), tf.size(un_3_223), tf.size(un_3_224), tf.size(un_3_225), tf.size(un_3_226), tf.size(un_3_227), tf.size(un_3_228), tf.size(un_3_229),\
    tf.size(un_3_230), tf.size(un_3_231), tf.size(un_3_232), tf.size(un_3_233), tf.size(un_3_234), tf.size(un_3_235), tf.size(un_3_236), tf.size(un_3_237), tf.size(un_3_238), tf.size(un_3_239),\
    tf.size(un_3_240), tf.size(un_3_241), tf.size(un_3_242), tf.size(un_3_243), tf.size(un_3_244), tf.size(un_3_245), tf.size(un_3_246), tf.size(un_3_247), tf.size(un_3_248), tf.size(un_3_249),\
    tf.size(un_3_250), tf.size(un_3_251), tf.size(un_3_252), tf.size(un_3_253), tf.size(un_3_254), tf.size(un_3_255), tf.size(un_3_256), tf.size(un_3_257), tf.size(un_3_258), tf.size(un_3_259),\
    tf.size(un_3_260), tf.size(un_3_261), tf.size(un_3_262), tf.size(un_3_263), tf.size(un_3_264), tf.size(un_3_265), tf.size(un_3_266), tf.size(un_3_267), tf.size(un_3_268), tf.size(un_3_269),\
    tf.size(un_3_270), tf.size(un_3_271), tf.size(un_3_272), tf.size(un_3_273), tf.size(un_3_274), tf.size(un_3_275), tf.size(un_3_276), tf.size(un_3_277), tf.size(un_3_278), tf.size(un_3_279),\
    tf.size(un_3_280), tf.size(un_3_281), tf.size(un_3_282), tf.size(un_3_283), tf.size(un_3_284), tf.size(un_3_285), tf.size(un_3_286), tf.size(un_3_287), tf.size(un_3_288), tf.size(un_3_289),\
    tf.size(un_3_290), tf.size(un_3_291), tf.size(un_3_292), tf.size(un_3_293), tf.size(un_3_294), tf.size(un_3_295), tf.size(un_3_296), tf.size(un_3_297), tf.size(un_3_298), tf.size(un_3_299),\
    tf.size(un_3_300), tf.size(un_3_301), tf.size(un_3_302), tf.size(un_3_303), tf.size(un_3_304), tf.size(un_3_305), tf.size(un_3_306), tf.size(un_3_307), tf.size(un_3_308), tf.size(un_3_309),\
    tf.size(un_3_310), tf.size(un_3_311), tf.size(un_3_312), tf.size(un_3_313), tf.size(un_3_314), tf.size(un_3_315), tf.size(un_3_316), tf.size(un_3_317), tf.size(un_3_318), tf.size(un_3_319),\
    tf.size(un_3_320), tf.size(un_3_321), tf.size(un_3_322), tf.size(un_3_323), tf.size(un_3_324), tf.size(un_3_325), tf.size(un_3_326), tf.size(un_3_327), tf.size(un_3_328), tf.size(un_3_329),\
    tf.size(un_3_330), tf.size(un_3_331), tf.size(un_3_332), tf.size(un_3_333), tf.size(un_3_334), tf.size(un_3_335), tf.size(un_3_336), tf.size(un_3_337), tf.size(un_3_338), tf.size(un_3_339),\
    tf.size(un_3_340), tf.size(un_3_341), tf.size(un_3_342), tf.size(un_3_343), tf.size(un_3_344), tf.size(un_3_345), tf.size(un_3_346), tf.size(un_3_347), tf.size(un_3_348), tf.size(un_3_349),\
    tf.size(un_3_350), tf.size(un_3_351), tf.size(un_3_352), tf.size(un_3_353), tf.size(un_3_354), tf.size(un_3_355), tf.size(un_3_356), tf.size(un_3_357), tf.size(un_3_358), tf.size(un_3_359),\
    tf.size(un_3_360), tf.size(un_3_361), tf.size(un_3_362), tf.size(un_3_363), tf.size(un_3_364), tf.size(un_3_365), tf.size(un_3_366), tf.size(un_3_367), tf.size(un_3_368), tf.size(un_3_369),\
    tf.size(un_3_370), tf.size(un_3_371), tf.size(un_3_372), tf.size(un_3_373), tf.size(un_3_374), tf.size(un_3_375), tf.size(un_3_376), tf.size(un_3_377), tf.size(un_3_378), tf.size(un_3_379),\
    tf.size(un_3_380), tf.size(un_3_381), tf.size(un_3_382), tf.size(un_3_383), tf.size(un_3_384), tf.size(un_3_385), tf.size(un_3_386), tf.size(un_3_387), tf.size(un_3_388), tf.size(un_3_389),\
    tf.size(un_3_390), tf.size(un_3_391), tf.size(un_3_392), tf.size(un_3_393), tf.size(un_3_394), tf.size(un_3_395), tf.size(un_3_396), tf.size(un_3_397), tf.size(un_3_398), tf.size(un_3_399),\
    tf.size(un_3_400), tf.size(un_3_401), tf.size(un_3_402), tf.size(un_3_403), tf.size(un_3_404), tf.size(un_3_405), tf.size(un_3_406), tf.size(un_3_407), tf.size(un_3_408), tf.size(un_3_409),\
    tf.size(un_3_410), tf.size(un_3_411), tf.size(un_3_412), tf.size(un_3_413), tf.size(un_3_414), tf.size(un_3_415), tf.size(un_3_416), tf.size(un_3_417), tf.size(un_3_418), tf.size(un_3_419),\
    tf.size(un_3_420), tf.size(un_3_421), tf.size(un_3_422), tf.size(un_3_423), tf.size(un_3_424), tf.size(un_3_425), tf.size(un_3_426), tf.size(un_3_427), tf.size(un_3_428), tf.size(un_3_429),\
    tf.size(un_3_430), tf.size(un_3_431), tf.size(un_3_432), tf.size(un_3_433), tf.size(un_3_434), tf.size(un_3_435), tf.size(un_3_436), tf.size(un_3_437), tf.size(un_3_438), tf.size(un_3_439),\
    tf.size(un_3_440), tf.size(un_3_441), tf.size(un_3_442), tf.size(un_3_443), tf.size(un_3_444), tf.size(un_3_445), tf.size(un_3_446), tf.size(un_3_447), tf.size(un_3_448), tf.size(un_3_449),\
    tf.size(un_3_450), tf.size(un_3_451), tf.size(un_3_452), tf.size(un_3_453), tf.size(un_3_454), tf.size(un_3_455), tf.size(un_3_456), tf.size(un_3_457), tf.size(un_3_458), tf.size(un_3_459),\
    tf.size(un_3_460), tf.size(un_3_461), tf.size(un_3_462), tf.size(un_3_463), tf.size(un_3_464), tf.size(un_3_465), tf.size(un_3_466), tf.size(un_3_467), tf.size(un_3_468), tf.size(un_3_469),\
    tf.size(un_3_470), tf.size(un_3_471), tf.size(un_3_472), tf.size(un_3_473), tf.size(un_3_474), tf.size(un_3_475), tf.size(un_3_476), tf.size(un_3_477), tf.size(un_3_478), tf.size(un_3_479),\
    tf.size(un_3_480), tf.size(un_3_481), tf.size(un_3_482), tf.size(un_3_483), tf.size(un_3_484), tf.size(un_3_485), tf.size(un_3_486), tf.size(un_3_487), tf.size(un_3_488), tf.size(un_3_489),\
    tf.size(un_3_490), tf.size(un_3_491), tf.size(un_3_492), tf.size(un_3_493), tf.size(un_3_494), tf.size(un_3_495), tf.size(un_3_496), tf.size(un_3_497), tf.size(un_3_498), tf.size(un_3_499)
    ]))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: #, allow_soft_placement=True
    #sess.run(tf.initialize_variables([total_max]))
    sess.run(iter.initializer)
    #best_result = 0
    #epochs = 100
    for i in range(100):
        if (i % 10 == 0):
            print(time.ctime(), i)
        #sess.run(max_op) #, feed_dict={Y: two_list})
        sess.run(all_max_0) #, feed_dict={Y: two_list})
            # if (result > best_result):
            #    best_result = result
    print(time.ctime())  # f'Best result: {result}')
        #if (result > best_result):
        #    best_result = result
    #print(f'Best result: {result}')