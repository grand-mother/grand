# Eclipse without installer

l'installer n'a pas fonctionn" sous docker, tarball de la dernière version 4.22 sans installaleur ici

[depot Eclipse](https://download.eclipse.org/eclipse/downloads/drops4/R-4.22-202111241800/)


# Eclipse configuration 

add marketplace (pour ajouter les plugging facilement
[tuto install marketplace](https://www.ibm.com/support/pages/how-install-marketplace-client-rational-performance-tester-eclipse-framework)

URL de la release utilisé 

http://download.eclipse.org/releases/photon/

ensuite:
- Egit
- PyDev
- YAML editor
- BASH editor

# Create docker image from container

[tuto image from container](https://www.sentinelone.com/blog/create-docker-image/)

```
docker commit <name container>
```
return ID IMAGE XXXid_imageYYY

```
docker tag XXXid_imageYYY name_of_image_from_container
```



