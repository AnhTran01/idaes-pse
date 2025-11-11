#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
General purpose unit selector block for IDAES models with GDP enforcing select exactly one
"""

# TODO: Add missing bounds to inlet/outlet

from enum import Enum
from functools import partial
from pandas import DataFrame

from pyomo.environ import TransformationFactory, Block
from pyomo.network import Port
from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    In,
    ListOf,
    Bool,
    ConfigDict,
    ConfigList,
)

from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    MaterialBalanceType,
    MomentumBalanceType,
    MaterialFlowBasis,
    VarLikeExpression,
)
from idaes.core.util.config import is_physical_parameter_block, is_state_block
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
    PropertyNotSupportedError,
    InitializationError,
)
from idaes.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.units_of_measurement import report_quantity
from idaes.core.initialization import ModularInitializerBase

from idaes.models.unit_models import Mixer, Separator

from pyomo.environ import RangeSet
from pyomo.network import Arc
from pyomo.gdp import Disjunct, Disjunction
from idaes.core.util.initialization import propagate_state

@declare_process_block_class("UnitSelectorV3")
class UnitSelectorData(UnitModelBlockData):
    default_initializer = None  # TBD

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
            **default** = False. Product blocks are always steady-state.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Product blocks do not contain holdup, thus this must be
            False.""",
        ),
    )

    CONFIG.declare(
        "unit_disjunct",
        ConfigValue(
            domain=list,
            description="Allocate the respective unit into the disjunction",
            doc="""WIP""",
        ),
    )

    CONFIG.declare(
        "unit_source",
        ConfigValue(
            description="",
            doc="""WIP""",
        ),
    )

    CONFIG.declare(
        "unit_sink",
        ConfigValue(
            description="",
            doc="""WIP""",
        ),
    )

    CONFIG.declare(
        "unit_disjunct_inlet",
        ConfigValue(
            description="",
            doc="""WIP""",
        ),
    )

    CONFIG.declare(
        "unit_disjunct_outlet",
        ConfigValue(
            description="",
            doc="""WIP""",
        ),
    )

    CONFIG.declare(
        "mixed_state_block",
        ConfigValue(
            domain=is_state_block,
            description="Existing StateBlock to use as mixed stream",
            doc="""An existing state block to use as the source stream from the
            Separator block,
            **default** - None.
            **Valid values:** {
            **None** - create a new StateBlock for the mixed stream,
            **StateBlock** - a StateBock to use as the source for the mixed stream.}""",
        ),
    )
    

    CONFIG.declare(
        "construct_ports",
        ConfigValue(
            default=True,
            domain=Bool,
            description="Construct inlet and outlet Port objects",
            doc="""Argument indicating whether model should construct Port
            objects linked to all inlet states and the mixed state,
            **default** - True.
            **Valid values:** {
            **True** - construct Ports for all states,
            **False** - do not construct Ports.""",
        ),
    )

    def build(self):
        """
        General build method for UnitSelectorData. This method calls a number
        of sub-methods which automate the construction of expected attributes
        of unit models.

        Inheriting models should call `super().build`.

        Args:
            None

        Returns:
            None
        """
        # Call super.build()
        super(UnitSelectorData, self).build()

        self.selector_block = Block()

        self.create_unit_disjunct()
        self.add_units_to_disjunct()

        # Create inlet list
        inlet_list = self.create_inlet_list()
        
        # Create outlet list
        outlet_list = self.create_outlet_list()

        self.selector_inlet_block = self.add_port_state_block(self.config.unit_source, port_type_list= inlet_list, is_inlet = True)
        self.selector_outlet_block = self.add_port_state_block(self.config.unit_sink, port_type_list= outlet_list, is_inlet = False)

        self.add_port_objects(inlet_list, is_inlet=True)
        self.add_port_objects(outlet_list, is_inlet=False)

        self.arc_selector_inlet_to_unit()
        self.arc_unit_outlet_to_selector()

        TransformationFactory("network.expand_arcs").apply_to(self)

    def create_unit_disjunct(self):
        for idx in RangeSet(len(self.config.unit_disjunct)):
            setattr(self.selector_block, f'unit_disjunct_{idx}', Disjunct())
        
        # Create the select only one disjunction
        setattr(
            self.selector_block,
            f'unit_disjunction',
            Disjunction(
                expr=[
                    getattr(self.selector_block, f'unit_disjunct_{idx}')
                    for idx in RangeSet(len(self.config.unit_disjunct))
                ]
            ),
        )

    def add_units_to_disjunct(self):
        selector_block = getattr(self, f'selector_block')
        units = self.config.unit_disjunct

        for unit_index, unit in enumerate(units):
            disjunct_obj = getattr(selector_block, f'unit_disjunct_{unit_index+1}')
            print('******', f'Moving unit: {unit.name} \nfrom parent block: {unit.parent_block()}')
            unit.parent_block().del_component(f'{unit.local_name}')
            disjunct_obj.add_component(f'{unit.local_name}', unit)
            print('******', f'to block: {disjunct_obj} \nas: {unit.name}\n')

    def create_inlet_list(self):
        """
        Create list of inlet stream names based on config arguments.

        Returns:
            list of strings
        """
        inlet_list = [
            "inlet_" + str(n) for n in range(1, len(self.config.unit_disjunct_inlet) + 1)
        ]

        return inlet_list

    def create_outlet_list(self):
        """
        Create list of outlet stream names based on config arguments.

        Returns:
            list of strings
        """
        outlet_list = [
            "outlet_" + str(n) for n in range(1, len(self.config.unit_disjunct_outlet) + 1)
        ]

        return outlet_list
    
    def add_port_state_block(self, unit_type, port_type_list, is_inlet = False):
        """
        Constructs StateBlock to represent inlet or outlet stream.

        Returns:
            New StateBlock object
        """
        # Setup StateBlock argument dict

        tmp_dict = dict(**unit_type.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["defined_state"] = True

        port_block = []

        direction = 'Inlet' if is_inlet else 'Outlet'

        for p in port_type_list:
            p_obj = unit_type.config.property_package.build_state_block(
                self.flowsheet().time, doc=f"Selector Unit {direction}", **tmp_dict
            )
            setattr(self, f'{p}_state', p_obj)
            port_block.append(getattr(self, f'{p}_state'))

        return port_block

    def add_port_objects(self, port_list, is_inlet = False):
        """
        Adds inlet Port object.

        Args:
            a mixed state StateBlock object

        Returns:
            None
        """
        direction = 'Inlet' if is_inlet else 'Outlet'

        if self.config.construct_ports is True:
            for p in port_list:
                p_state = getattr(self, f'{p}_state')
                self.add_port(name=f'{p}', block=p_state, doc=f"Connector {direction} Port")
    
    def update_port_bounds(self):
        """
        Update the inlet/outlet port of the selector based on the unit_source/unit_sink provided
        """
        pass

    def initialize(self, **kwargs):
        selector_block = getattr(self, f'selector_block')
        units = self.config.unit_disjunct

        for unit_index, unit in enumerate(units):
            idx = unit_index + 1 
            disjunct_block = getattr(selector_block, f'unit_disjunct_{idx}')

            unit_block = getattr(disjunct_block, unit.local_name)

            inlet_arc = getattr(disjunct_block, f'inlet_to_unit_arc_{idx}')

            print(f'Propagate to inlet arc: {inlet_arc}')
            propagate_state(inlet_arc)

            print(f'Initialize unit {unit.name}')
            unit_block.initialize(**kwargs)

            outlet_arc = getattr(disjunct_block, f'unit_to_outlet_arc_{idx}')
            print(f'Propagate to outlet arc: {outlet_arc}\n')

    def arc_selector_inlet_to_unit(self):
        for inlet_idx, unit_inlet_list in self.config.unit_disjunct_inlet.items():
            
            if len(unit_inlet_list) > 1:
                dsj_counter = 1
                for unit_inlet_port in unit_inlet_list:
                    unit_disjunct = getattr(self.selector_block, f'unit_disjunct_{dsj_counter}')
                    selector_inlet_port = getattr(self, f'inlet_{inlet_idx}')

                    unit_disjunct.add_component(
                        f"inlet_to_unit_arc_{dsj_counter}",
                        Arc(source=selector_inlet_port, destination=unit_inlet_port),
                    )
                    dsj_counter += 1
            
            # For single-element list, omit looping
            else:
                unit_disjunct = getattr(self.selector_block, f'unit_disjunct_{inlet_idx}')
                selector_inlet_port = getattr(self, f'inlet_{inlet_idx}')
                unit_disjunct.add_component(
                    f"inlet_to_unit_arc_{inlet_idx}",
                    Arc(source=selector_inlet_port, destination=unit_inlet_list[0]),
                )
    def arc_unit_outlet_to_selector(self):
        for outlet_idx, unit_outlet_list in self.config.unit_disjunct_outlet.items():
            
            if len(unit_outlet_list) > 1:
                dsj_counter = 1
                for unit_outlet_port in unit_outlet_list:
                    unit_disjunct = getattr(self.selector_block, f'unit_disjunct_{dsj_counter}')
                    selector_outlet_port = getattr(self, f'outlet_{outlet_idx}')

                    unit_disjunct.add_component(
                        f"unit_to_outlet_arc_{dsj_counter}",
                        Arc(source=unit_outlet_port, destination=selector_outlet_port),
                    )
                    dsj_counter += 1
            
            # For single-element list, omit looping
            else:
                unit_disjunct = getattr(self.selector_block, f'unit_disjunct_{outlet_idx}')
                selector_outlet_port = getattr(self, f'outlet_{outlet_idx}')
                unit_disjunct.add_component(
                    f"unit_to_outlet_arc_{outlet_idx}",
                    Arc(source=unit_outlet_list[0], destination=selector_outlet_port),
                )