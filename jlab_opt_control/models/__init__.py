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

from jlab_opt_control.models.actor_fcnn import ActorFCNN
from jlab_opt_control.models.actor_fcnn_v2 import ActorFCNN_v2
from jlab_opt_control.models.actor_gaussian_v0 import ActorGaussian
from jlab_opt_control.models.critic_fcnn import CriticFCNN
from jlab_opt_control.models.critic_fcnn_v1 import CriticFCNN_v1
from jlab_opt_control.models.critic_fcnn_v2 import CriticFCNN_v2
from jlab_opt_control.models.registration import register, make, list_registered_modules
from jlab_opt_control.models.sindy_network import SINDyNetwork
from jlab_opt_control.models.uqsindy_network import UQSINDyNetwork

register(
    id='actor_fcnn-v0',
    entry_point='jlab_opt_control.models:ActorFCNN',
    kwargs={'cfg': 'actor_fcnn.cfg'},
)

register(
    id='actor_gaussian-v0',
    entry_point='jlab_opt_control.models:ActorGaussian',
    kwargs={'cfg': 'actor_fcnn.cfg'},
)

register(
    id='critic_fcnn-v0',
    entry_point='jlab_opt_control.models:CriticFCNN',
    kwargs={'cfg': 'critic_fcnn.cfg'},
)

register(
    id='actor_fcnn-v2',
    entry_point='jlab_opt_control.models:ActorFCNN_v2',
    kwargs={'cfg': 'actor_fcnn.cfg'},
)

register(
    id='critic_fcnn-v1',
    entry_point='jlab_opt_control.models:CriticFCNN_v1',
    kwargs={'cfg': 'critic_fcnn_v1.cfg'},
)

register(
    id='critic_fcnn-v2',
    entry_point='jlab_opt_control.models:CriticFCNN_v2',
    kwargs={'cfg': 'critic_fcnn_v2.cfg'},
)


register(
    id='sindy_network-v0',
    entry_point='jlab_opt_control.models:SINDyNetwork',
    kwargs={'cfg': 'sindy_network.cfg'}
)

register(
    id='uqsindy_network-v0',
    entry_point='jlab_opt_control.models:UQSINDyNetwork',
    kwargs={'cfg': 'uqsindy_network.cfg'}
)

from jlab_opt_control.models.mo_actor_fcnn import MO_ActorFCNN
register(
    id="mo_actor_fcnn-v0",
    entry_point='jlab_opt_control.models:MO_ActorFCNN',
    kwargs={'cfg': 'actor_fcnn.cfg'}
)

from jlab_opt_control.models.mo_actor_fcnn_v2 import MO_ActorFCNN_v2
register(
    id="mo_actor_fcnn-v2",
    entry_point='jlab_opt_control.models:MO_ActorFCNN_v2',
    kwargs={'cfg': 'actor_fcnn.cfg'}
)

from jlab_opt_control.models.mo_critic_fcnn import MO_CriticFCNN
register(
    id="mo_critic_fcnn-v0",
    entry_point='jlab_opt_control.models:MO_CriticFCNN',
    kwargs={'cfg': 'critic_fcnn.cfg'}
)

from jlab_opt_control.models.mo_stateless_actor_fcnn import MO_StatelessActorFCNN
from jlab_opt_control.models.mo_stateless_critic_fcnn import MO_StatelessCriticFCNN
register(
    id="mo_stateless_actor_fcnn-v0",
    entry_point='jlab_opt_control.models:MO_StatelessActorFCNN',
    kwargs={'cfg': 'actor_fcnn.cfg'}
)

register(
    id="mo_stateless_critic_fcnn-v0",
    entry_point='jlab_opt_control.models:MO_StatelessCriticFCNN',
    kwargs={'cfg': 'critic_fcnn.cfg'}
)

from jlab_opt_control.models.mo_actor_fcnn_cic import MO_ActorFCNN_CIC
register(
    id="mo_actor_fcnn-cic-v0",
    entry_point='jlab_opt_control.models:MO_ActorFCNN_CIC',
    kwargs={'cfg': 'actor_fcnn.cfg'}
)