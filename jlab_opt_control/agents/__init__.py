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

from jlab_opt_control.agents.registration import register, make, list_registered_modules
from jlab_opt_control.agents.keras_td3 import KerasTD3
from jlab_opt_control.agents.keras_uncertainty_td3 import KerasUncertaintyTD3
from jlab_opt_control.agents.keras_ddpg import KerasDDPG
from jlab_opt_control.agents.keras_sindy_td3 import KerasSINDyTD3
from jlab_opt_control.agents.keras_sindy_critic_td3 import KerasSINDyCriticTD3
from jlab_opt_control.agents.keras_joint_critic_td3 import KerasJointCriticTD3

# Single Objective Agents
register(
    id="KerasTD3-v0",
    entry_point="jlab_opt_control.agents:KerasTD3",
    kwargs={"cfg": "keras_td3.cfg"},
)

register(
    id="KerasUncertaintyTD3-v0",
    entry_point="jlab_opt_control.agents:KerasUncertaintyTD3",
    kwargs={"cfg": "keras_uncertainty_td3.cfg"},
)

register(
    id="KerasDDPG-v0",
    entry_point="jlab_opt_control.agents:KerasDDPG",
    kwargs={"cfg": "keras_ddpg.cfg"},
)

register(
    id="KerasSINDyCriticTD3-v0",
    entry_point="jlab_opt_control.agents:KerasSINDyCriticTD3",
    kwargs={"cfg": "keras_sindy_td3.cfg"},
)

register(
    id="KerasJointCriticTD3-v0",
    entry_point="jlab_opt_control.agents:KerasJointCriticTD3",
    kwargs={"cfg": "keras_joint_td3.cfg"},
)

register(
    id="KerasSINDyTD3-v0",
    entry_point="jlab_opt_control.agents:KerasSINDyTD3",
    kwargs={"cfg": "keras_td3.cfg"},
)
