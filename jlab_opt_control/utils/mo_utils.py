# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Kishan Rajput (kishan@jlab.org)

import numpy as np

def get_fraction_mono_tuning(heat, trip, alphas, debug=False):
    if len(alphas) <= 0:
        return 0.0
    sorted_indices = np.argsort(alphas)
    sorted_alphas = alphas[sorted_indices]
    sorted_heat = heat[sorted_indices]
    sorted_trip = trip[sorted_indices]
    counter = 0
    for i, a in enumerate(sorted_alphas):
        if i == 0:
            continue
        if sorted_heat[i] < sorted_heat[i-1] or sorted_trip[i] > sorted_trip[i-1]:
            counter += 1
            if debug:
                print("Counter: ", counter)
                print("index: ", i-1, " alpha: ", sorted_alphas[i-1], " heat: ", sorted_heat[i-1], " trip: ", sorted_trip[i-1])
                print("index: ", i, " alpha: ", sorted_alphas[i], " heat: ", sorted_heat[i], " trip: ", sorted_trip[i])
                print("################################################################################")
    percentage_mono = np.round((1 - counter / len(sorted_alphas))*100, 3)
    return percentage_mono
