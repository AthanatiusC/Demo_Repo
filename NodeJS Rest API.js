let db = require('../config/connection')

exports.index = function(req, res){
    db.query("SELECT * FROM users", function(err,result){
    	if(err) res.status(500).send({'result':err})

    		res.status(200).send({'result':result})
    })
}
exports.detail = function(req, res){
    let id = req.params.id
    db.query("SELECT * FROM users WHERE id=?", id, function(err, result){
    	if(err) res.status(500).send({'result':err})

    		res.status(200).send({'result':result})
    })

}
exports.store = function(req, res){
    let data = {
    	name : req.body.name,
    	email : req.body.email,
    	phone : req.body.phone,
    	address : req.body.address,
    	password : req.body.password,
    	profile_image : req.body.profile_image,
    	level : req.body.level,
    }
    db.query('INSERT INTO users (name, email, phone, address, password, profile_image, level) VALUES (?,?,?,?,?,?,?)',
    	[data.name, data.email, data.phone, data.address, data.password, data.profile_image, data.level], function(error,results){
    	if (error) throw error;

    	let id = results.insertId;
    	if (id) {
    		data.id == results.insertId;
    		res.send(data,201)
    	}else{
    		res.send(400)
    	}

})
}
exports.update = function (req, res) {
    let id = req.params.id
    let data = {
        name : req.body.name,
    	email : req.body.email,
    	phone : req.body.phone,
    	address : req.body.address,
    	password : req.body.password,
    	profile_image : req.body.profile_image,
    	level : req.body.level,
    }
    db.query('UPDATE users set name=?, email=?, phone=?, address=?, password=?, profile_image=?, level=? WHERE id=?',
        [data.name, data.email, data.phone, data.address, data.password, data.profile_image, data.level, id],
        function (error, results) {
            if (error) throw error
            let changedRows = results.changedRows;
            if (changedRows > 0) {
                data.id = id
                res.send(data, 200)
            } else {
                res.send(404)
            }
        })
}
exports.delete = function(req, res){
    var id = req.params.id
    db.query('DELETE FROM users WHERE id=?' ,[id], function(error,result){
    	if (error) throw error;

    	if(result.affectedRows > 0){
    		res.send(204)
    	}else{
    		res.send(404)
    	}
    })
}