from qgis.core import QgsApplication

qgs = QgsApplication([], False)
qgs.initQgis()

from qgis.core import QgsProject
print("QGIS loaded OK")
print("Project CRS:", QgsProject.instance().crs().authid())

qgs.exitQgis()