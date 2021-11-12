# -*- coding: utf-8 -*-
"""
/***************************************************************************
 classe
                                 A QGIS plugin
 susceptibility
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-07-01
        copyright            : (C) 2021 by Giacomo Titti
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load classe class from file classe.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .sz_module import classePlugin
    return classePlugin()
